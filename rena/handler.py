"""Pure `Request -> Response` handler.

Owns image preprocessing, batch assembly, waypoint selection, and the PD
controller that turns the model's waypoint chunk into a `cmd_vel`. The
handler never sees a ROS message, HTTP request, or gRPC stub — those live in
the transport layer.

Mirrors the logic in `inference/run_omnivla_edge.py` so the same algorithm
runs whether you exercise it via the one-shot script or via the RENA service.
"""

from __future__ import annotations

import io
import math
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from .messages import NavVlaRequest, NavVlaResponse
from .model import OmniVlaModel

# Same flat imports as model.py — the helpers live in inference/.
_OMNIVLA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_OMNIVLA_ROOT, os.path.join(_OMNIVLA_ROOT, "inference")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils_policy import transform_images_PIL_mask, transform_images_map  # noqa: E402


_IMG_SIZE = (96, 96)
_LARGE_IMG_SIZE = (224, 224)
_SAT_IMG_SIZE = (352, 352)


@dataclass
class HandlerConfig:
    """Knobs the handler applies to every inference call."""

    metric_waypoint_spacing: float = 0.1
    waypoint_select: int = 4               # which waypoint in the chunk drives cmd_vel
    control_dt_s: float = 1.0 / 3.0        # 3 Hz tick
    max_linear: float = 0.3
    max_angular: float = 0.3


def _clip_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


class NavVlaHandler:
    """Stateless handler. Construct once with a loaded model + config."""

    def __init__(self, model: OmniVlaModel, cfg: HandlerConfig | None = None) -> None:
        self._model = model
        self._cfg = cfg or HandlerConfig()
        # Plain unit masks — fisheye-specific masking can be plugged in here
        # if needed (see the commented block in run_omnivla_edge.py).
        self._mask_96 = np.ones((96, 96, 3), dtype=np.float32)
        self._mask_224 = np.ones((224, 224, 3), dtype=np.float32)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def __call__(self, req: NavVlaRequest) -> NavVlaResponse:
        batch = self._build_batch(req)
        actions, modality_used = self._model.forward(**batch)
        waypoints = actions.float().cpu().numpy()[0]   # [chunk_len, 4]

        linear, angular = self._waypoints_to_cmd(waypoints)

        return NavVlaResponse(
            stamp_ns=req.stamp_ns,
            linear=linear,
            angular=angular,
            waypoint_x=waypoints[:, 0].tolist(),
            waypoint_y=waypoints[:, 1].tolist(),
            waypoint_cos=waypoints[:, 2].tolist(),
            waypoint_sin=waypoints[:, 3].tolist(),
            modality_id_used=int(modality_used.cpu().numpy().reshape(-1)[0]),
        )

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    def _build_batch(self, req: NavVlaRequest) -> dict[str, torch.Tensor]:
        device = self._model.device
        if len(req.obs_jpegs) == 0:
            raise ValueError("NavVlaRequest.obs_jpegs must contain at least one frame")

        # Observation history (96x96, six frames). If fewer were supplied (e.g.
        # the robot just started), pad by repeating the oldest available frame.
        ctx_size = 6
        pil_obs = [_decode_jpeg(b).resize(_IMG_SIZE) for b in req.obs_jpegs]
        while len(pil_obs) < ctx_size:
            pil_obs.insert(0, pil_obs[0])
        pil_obs = pil_obs[-ctx_size:]

        obs_tensor = transform_images_PIL_mask(pil_obs, self._mask_96).to(device)
        # Split into individual frames so we can isolate the current one for
        # the satellite/cur fusion below.
        per_frame = torch.split(obs_tensor, 3, dim=1)
        obs_cur = per_frame[-1].to(device)
        obs_images = torch.cat(per_frame, dim=1).to(device)

        # Current frame at 224x224 for the CLIP-visual branch.
        cur_large_pil = _decode_jpeg(req.cur_large_jpeg).resize(_LARGE_IMG_SIZE)
        cur_large = transform_images_PIL_mask([cur_large_pil], self._mask_224).to(device)

        # Goal pose in the robot frame (x forward, y left, yaw CCW). The model
        # was trained on the same convention: `run_omnivla_edge.py` builds the
        # pose vector as [rel_y / s, -rel_x / s, cos, sin], where its rel_x is
        # "right of heading" and rel_y is "along heading", so after the
        # negation `pose[1]` is "left / s" — what we already have.
        s = self._cfg.metric_waypoint_spacing
        goal_pose = torch.tensor(
            [req.goal_x / s, req.goal_y / s, math.cos(req.goal_yaw), math.sin(req.goal_yaw)],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        # Goal image — fall back to the current obs if the caller didn't supply
        # one. With modality_id=4 (pose-only) this branch is masked out anyway.
        goal_pil = (
            _decode_jpeg(req.goal_jpeg).resize(_IMG_SIZE)
            if req.goal_jpeg
            else pil_obs[-1]
        )
        goal_image = transform_images_PIL_mask([goal_pil], self._mask_96).to(device)

        # Dummy satellite tiles — we don't have GPS / aerial imagery on the robot.
        sat_blank = Image.new("RGB", _SAT_IMG_SIZE, color=(0, 0, 0))
        sat_cur = transform_images_map(sat_blank).to(device)
        sat_goal = transform_images_map(sat_blank).to(device)
        map_images = torch.cat((sat_cur, sat_goal, obs_cur), dim=1)

        language_tokens = self._model.tokenize(req.language_prompt)
        modality_id = torch.tensor([req.modality_id], device=device)

        return {
            "obs_images": obs_images,
            "cur_large": cur_large,
            "goal_pose": goal_pose,
            "goal_image": goal_image,
            "map_images": map_images,
            "language_tokens": language_tokens,
            "modality_id": modality_id,
        }

    # ------------------------------------------------------------------
    # Waypoint -> cmd_vel (PD controller, ported from run_omnivla_edge.py)
    # ------------------------------------------------------------------
    def _waypoints_to_cmd(self, waypoints: np.ndarray) -> tuple[float, float]:
        idx = min(self._cfg.waypoint_select, len(waypoints) - 1)
        chosen = waypoints[idx].copy()
        chosen[:2] *= self._cfg.metric_waypoint_spacing
        dx, dy, hx, hy = chosen
        dt = self._cfg.control_dt_s
        eps = 1e-8

        if abs(dx) < eps and abs(dy) < eps:
            linear = 0.0
            angular = _clip_angle(math.atan2(hy, hx)) / dt
        elif abs(dx) < eps:
            linear = 0.0
            angular = np.sign(dy) * math.pi / (2 * dt)
        else:
            linear = dx / dt
            angular = math.atan(dy / dx) / dt

        linear = float(np.clip(linear, 0.0, 0.5))
        angular = float(np.clip(angular, -1.0, 1.0))

        return self._limit_velocity(linear, angular)

    def _limit_velocity(self, linear: float, angular: float) -> tuple[float, float]:
        maxv, maxw = self._cfg.max_linear, self._cfg.max_angular
        if abs(linear) <= maxv and abs(angular) <= maxw:
            return linear, angular
        if abs(linear) <= maxv:
            # Angular over-limit: scale linear by the ratio.
            rd = linear / angular if angular else 0.0
            return maxw * np.sign(linear) * abs(rd), maxw * np.sign(angular)
        if abs(angular) <= 1e-3:
            return maxv * np.sign(linear), 0.0
        rd = linear / angular
        if abs(rd) >= maxv / maxw:
            return maxv * np.sign(linear), maxv * np.sign(angular) / abs(rd)
        return maxw * np.sign(linear) * abs(rd), maxw * np.sign(angular)


def _decode_jpeg(buf: bytes) -> Image.Image:
    return Image.open(io.BytesIO(buf)).convert("RGB")
