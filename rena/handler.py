"""Pure `NavVlaRequest -> NavVlaResponse` handler.

Model-agnostic: it decodes JPEGs to PIL, hands raw frames + goal pose to
`model.predict(...)`, then runs the PD controller / velocity limiter on the
returned waypoint chunk.

The actual preprocessing (image transforms, prompt building, tokenization,
etc.) lives inside each model wrapper so the handler stays the same whether
you're running OmniVLA-edge or the full OpenVLA-OFT family.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .messages import NavVlaRequest, NavVlaResponse
from .model import OmniVlaModelProto


@dataclass
class HandlerConfig:
    """Knobs the handler applies to every inference call."""

    metric_waypoint_spacing: float = 0.1
    waypoint_select: int = 4                # which waypoint in the chunk drives cmd_vel
    control_dt_s: float = 1.0 / 3.0         # 3 Hz tick
    max_linear: float = 0.35                # m/s   (nav2 MPPI baseline)
    max_angular: float = 0.4                # rad/s (nav2 MPPI baseline)


def _clip_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class NavVlaHandler:
    """Stateless handler. Construct once with a loaded model + config."""

    def __init__(self, model: OmniVlaModelProto, cfg: HandlerConfig | None = None) -> None:
        self._model = model
        self._cfg = cfg or HandlerConfig()

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def __call__(self, req: NavVlaRequest) -> NavVlaResponse:
        if len(req.obs_jpegs) == 0:
            raise ValueError("NavVlaRequest.obs_jpegs must contain at least one frame")

        obs_pils = [_decode_jpeg(b) for b in req.obs_jpegs]
        cur_pil = _decode_jpeg(req.cur_large_jpeg)
        goal_pil = _decode_jpeg(req.goal_jpeg) if req.goal_jpeg else None

        actions, modality_used = self._model.predict(
            obs_pils=obs_pils,
            cur_pil=cur_pil,
            goal_pil=goal_pil,
            goal_pose=(req.goal_x, req.goal_y, req.goal_yaw),
            language_prompt=req.language_prompt,
            modality_id=req.modality_id,
        )
        waypoints = actions.float().cpu().numpy()[0]   # [chunk, 4]

        linear, angular = self._waypoints_to_cmd(waypoints)

        return NavVlaResponse(
            stamp_ns=req.stamp_ns,
            linear=linear,
            angular=angular,
            waypoint_x=waypoints[:, 0].tolist(),
            waypoint_y=waypoints[:, 1].tolist(),
            waypoint_cos=waypoints[:, 2].tolist(),
            waypoint_sin=waypoints[:, 3].tolist(),
            modality_id_used=int(modality_used),
        )

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
