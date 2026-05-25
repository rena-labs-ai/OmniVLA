"""OmniVLA model wrappers.

Two model variants share the same `predict()` signature so the handler doesn't
need to know which one it has:
  * `OmniVlaEdgeModel` — the small EfficientNet-b0 + transformer-head model
    (`omnivla-edge`).
  * `OmniVlaModel`     — the full OpenVLA-OFT model
    (`omnivla-original`, `omnivla-original-balance`, `omnivla-finetuned-cast`).

Each class owns its full preprocessing pipeline — JPEG-decoded PIL frames go
in, predicted waypoint chunk comes out. The handler is the same for both.

Switch via `model.name` in `config.yaml`.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch
from PIL import Image as PILImage

# The inference/ helpers (`load_model`, `transform_*`) are imported as flat
# modules — add them to sys.path the same way `run_omnivla_edge.py` does.
_OMNIVLA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_OMNIVLA_ROOT, os.path.join(_OMNIVLA_ROOT, "inference")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import clip  # noqa: E402
from utils_policy import (  # noqa: E402
    load_model,
    transform_images_PIL_mask,
    transform_images_map,
)


# Each name doubles as the HF repo (NHirose/<name>) and the directory name
# under the OmniVLA repo root. Adding a new checkpoint = one entry here.
EDGE_NAMES = ("omnivla-edge",)
FULL_NAMES = ("omnivla-original", "omnivla-original-balance", "omnivla-finetuned-cast")
VALID_NAMES = EDGE_NAMES + FULL_NAMES

# Per-checkpoint resume step (used by the full OmniVLA loader to find the
# `pose_projector--<step>_checkpoint.pt` and `action_head--<step>_checkpoint.pt`
# files inside the HF repo).
_FULL_RESUME_STEPS = {
    "omnivla-original": 120000,
    "omnivla-original-balance": 120000,
    "omnivla-finetuned-cast": 210000,
}


# ---- Config ------------------------------------------------------------------
@dataclass
class ModelConfig:
    """The only knob is the checkpoint name; everything else is derived."""

    name: str
    device: str = "cuda:0"


# Fixed architecture params for the OmniVLA-edge checkpoint. Tied to the
# trained weights, not user-tunable — they live next to the loader.
_EDGE_ARCH = {
    "model_type": "omnivla-edge",
    "len_traj_pred": 8,
    "context_size": 5,
    "learn_angle": True,
    "obs_encoder": "efficientnet-b0",
    "encoding_size": 256,
    "obs_encoding_size": 1024,
    "goal_encoding_size": 1024,
    "late_fusion": False,
    "mha_num_attention_heads": 4,
    "mha_num_attention_layers": 4,
    "mha_ff_dim_factor": 4,
    "clip_type": "ViT-B/32",
}

_OBS_SIZE = (96, 96)
_LARGE_SIZE = (224, 224)
_SAT_SIZE = (352, 352)
_OBS_HISTORY_LEN = 6
# Number of images the full OmniVLA's vision backbone consumes per request
# (current + goal — matches `InferenceConfig.num_images_in_input` in
# inference/run_omnivla.py).
_FULL_NUM_IMAGES = 2


def _weights_path_for(name: str) -> str:
    """Resolve `<OmniVLA-root>/<name>/<name>.pth` for edge checkpoints, or
    `<OmniVLA-root>/<name>/` for full OmniVLA HF repos."""
    base = os.path.join(_OMNIVLA_ROOT, name)
    if name in EDGE_NAMES:
        return os.path.join(base, f"{name}.pth")
    return base


# ---- Common interface --------------------------------------------------------
@runtime_checkable
class OmniVlaModelProto(Protocol):
    """Duck-typed shape both model wrappers expose."""

    @property
    def device(self) -> torch.device: ...

    def predict(
        self,
        obs_pils: list[PILImage.Image],
        cur_pil: PILImage.Image,
        goal_pil: PILImage.Image | None,
        goal_pose: tuple[float, float, float],   # (x, y, yaw) in robot frame
        language_prompt: str,
        modality_id: int,
    ) -> tuple[torch.Tensor, int]:
        """Returns (predicted_actions [B, chunk, 4], modality_id_used)."""
        ...


# ---- Helpers shared by both wrappers ----------------------------------------
def _build_edge_obs_history(
    obs_pils: list[PILImage.Image], mask_96: np.ndarray, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resize, mask-norm, and concat the observation history (96x96 frames).
    Returns (obs_images_concat, obs_image_cur) — both on `device`.
    """
    if not obs_pils:
        raise ValueError("obs_pils must contain at least one frame")
    frames = [p.resize(_OBS_SIZE) for p in obs_pils]
    while len(frames) < _OBS_HISTORY_LEN:
        frames.insert(0, frames[0])
    frames = frames[-_OBS_HISTORY_LEN:]
    tensor = transform_images_PIL_mask(frames, mask_96).to(device)
    per_frame = torch.split(tensor, 3, dim=1)
    obs_cur = per_frame[-1].to(device)
    return torch.cat(per_frame, dim=1).to(device), obs_cur


# ---- OmniVLA-edge wrapper ---------------------------------------------------
class OmniVlaEdgeModel:
    """Loaded OmniVLA-edge model + CLIP text encoder.

    Preprocessing (resize, mask, transform) happens inside `predict()`, so the
    handler doesn't need to know about it.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        path = _weights_path_for(cfg.name)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"OmniVLA-edge weights not found: {path}\n"
                f"Download with:  git lfs install && "
                f"git clone https://huggingface.co/NHirose/{cfg.name} "
                f"{os.path.dirname(path)}"
            )
        self._device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        model, text_encoder, _preprocess = load_model(path, _EDGE_ARCH, self._device)
        self._model = model.to(self._device).eval()
        self._text_encoder = text_encoder.to(self._device).eval()

        # Unit masks — fisheye-specific masks can be plugged in here.
        self._mask_96 = np.ones((96, 96, 3), dtype=np.float32)
        self._mask_224 = np.ones((224, 224, 3), dtype=np.float32)
        # Spacing used to normalize goal pose for the model.
        self._spacing = 0.1

    @property
    def device(self) -> torch.device:
        return self._device

    @torch.no_grad()
    def predict(
        self,
        obs_pils: list[PILImage.Image],
        cur_pil: PILImage.Image,
        goal_pil: PILImage.Image | None,
        goal_pose: tuple[float, float, float],
        language_prompt: str,
        modality_id: int,
    ) -> tuple[torch.Tensor, int]:
        # Obs history (6 frames at 96x96).
        obs_images, obs_cur = _build_edge_obs_history(obs_pils, self._mask_96, self._device)

        # Current frame at 224x224 for the CLIP-visual branch.
        cur_large = transform_images_PIL_mask(
            [cur_pil.resize(_LARGE_SIZE)], self._mask_224
        ).to(self._device)

        # Goal pose — convention (x=forward, y=left, yaw=CCW) matches Rena.
        gx, gy, gyaw = goal_pose
        s = self._spacing
        goal_pose_t = torch.tensor(
            [gx / s, gy / s, math.cos(gyaw), math.sin(gyaw)],
            dtype=torch.float32, device=self._device,
        ).unsqueeze(0)

        # Fall back to the latest obs if no goal image (modality 4 ignores it
        # but the model still needs a tensor of the right shape).
        goal_src = goal_pil if goal_pil is not None else obs_pils[-1]
        goal_image = transform_images_PIL_mask(
            [goal_src.resize(_OBS_SIZE)], self._mask_96
        ).to(self._device)

        # Dummy satellite tiles — no aerial imagery on the robot.
        sat_blank = PILImage.new("RGB", _SAT_SIZE, color=(0, 0, 0))
        sat_cur = transform_images_map(sat_blank).to(self._device)
        sat_goal = transform_images_map(sat_blank).to(self._device)
        map_images = torch.cat((sat_cur, sat_goal, obs_cur), dim=1)

        # CLIP text.
        tokens = clip.tokenize(language_prompt or "xxxx", truncate=True).to(self._device)
        feat_text = self._text_encoder.encode_text(tokens)

        modality_t = torch.tensor([modality_id], device=self._device)
        bimg = goal_image.size(0)
        predicted_actions, _distances, mask_number = self._model(
            obs_images.repeat(bimg, 1, 1, 1),
            goal_pose_t.repeat(bimg, 1),
            map_images.repeat(bimg, 1, 1, 1),
            goal_image,
            modality_t.repeat(bimg),
            feat_text.repeat(bimg, 1),
            cur_large.repeat(bimg, 1, 1, 1),
        )
        return predicted_actions, int(mask_number.cpu().numpy().reshape(-1)[0])


# ---- Full OmniVLA wrapper ---------------------------------------------------
class OmniVlaModel:
    """Loaded full OmniVLA (OpenVLA-OFT) model.

    Mirrors `inference/run_omnivla.py:define_model` for loading and
    `run_forward_pass` for inference. The class owns its preprocessing
    (prompt building + image processor + tokenizer + collator).
    """

    def __init__(self, cfg: ModelConfig) -> None:
        path = _weights_path_for(cfg.name)
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"OmniVLA weights directory not found: {path}\n"
                f"Download with:  git lfs install && "
                f"git clone https://huggingface.co/NHirose/{cfg.name} {path}"
            )
        if cfg.name not in _FULL_RESUME_STEPS:
            raise ValueError(f"no resume_step recorded for {cfg.name!r}")
        step = _FULL_RESUME_STEPS[cfg.name]

        # Heavy deps imported here so the edge variant doesn't pay for them.
        from transformers import (
            AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor,
        )
        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
        from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction_MMNv1
        from prismatic.extern.hf.processing_prismatic import (
            PrismaticImageProcessor, PrismaticProcessor,
        )
        from prismatic.models.action_heads import L1RegressionActionHead_idcat
        from prismatic.models.backbones.llm.prompting import PurePromptBuilder
        from prismatic.models.projectors import ProprioProjector
        from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
        from prismatic.vla.action_tokenizer import ActionTokenizer
        from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, POSE_DIM

        device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()

        # Register OpenVLA classes with HF Auto* (idempotent — no-op on re-runs).
        try:
            AutoConfig.register("openvla", OpenVLAConfig)
            AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
            AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
            AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction_MMNv1)
        except ValueError:
            pass  # Already registered in this process.

        processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        vla = AutoModelForVision2Seq.from_pretrained(
            path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        ).to(device)
        vla.vision_backbone.set_num_images_in_input(_FULL_NUM_IMAGES)
        vla.to(dtype=torch.bfloat16, device=device).eval()

        pose_projector = ProprioProjector(llm_dim=vla.llm_dim, proprio_dim=POSE_DIM)
        pose_projector.load_state_dict(
            _load_full_ckpt("pose_projector", path, step, device)
        )
        pose_projector = pose_projector.to(device).eval()

        action_head = L1RegressionActionHead_idcat(
            input_dim=vla.llm_dim, hidden_dim=vla.llm_dim, action_dim=ACTION_DIM,
        )
        action_head.load_state_dict(
            _load_full_ckpt("action_head", path, step, device)
        )
        action_head = action_head.to(torch.bfloat16).to(device).eval()

        num_patches = (
            vla.vision_backbone.get_num_patches()
            * vla.vision_backbone.get_num_images_in_input()
            + 1  # +1 for goal pose
        )

        self._device = device
        self._vla = vla
        self._processor = processor
        self._pose_projector = pose_projector
        self._action_head = action_head
        self._action_tokenizer = ActionTokenizer(processor.tokenizer)
        self._prompt_builder_cls = PurePromptBuilder
        self._num_patches = num_patches
        self._action_dim = ACTION_DIM
        self._chunk_len = NUM_ACTIONS_CHUNK
        self._spacing = 0.1
        self._get_current_action_mask = get_current_action_mask
        self._get_next_actions_mask = get_next_actions_mask

    @property
    def device(self) -> torch.device:
        return self._device

    @torch.no_grad()
    def predict(
        self,
        obs_pils: list[PILImage.Image],
        cur_pil: PILImage.Image,
        goal_pil: PILImage.Image | None,
        goal_pose: tuple[float, float, float],
        language_prompt: str,
        modality_id: int,
    ) -> tuple[torch.Tensor, int]:
        from prismatic.vla.constants import IGNORE_INDEX

        # Goal pose vector (x_fwd/s, y_left/s, cos(yaw), sin(yaw)).
        gx, gy, gyaw = goal_pose
        s = self._spacing
        goal_pose_t = torch.tensor(
            [gx / s, gy / s, math.cos(gyaw), math.sin(gyaw)],
            dtype=torch.float32,
        ).unsqueeze(0)  # [1, 4]

        # The full model expects two images (current + goal) — fall back to the
        # current frame as the goal when no image goal was provided.
        cur224 = cur_pil.resize(_LARGE_SIZE)
        goal224 = (goal_pil or cur_pil).resize(_LARGE_SIZE)

        # Build prompt + tokenize. The action chunk string is a placeholder
        # whose length sets the slot for the predicted-action hidden states.
        actions_dummy = np.random.rand(self._chunk_len, self._action_dim)
        action_string = "".join(self._action_tokenizer(actions_dummy[1:]))
        action_string = self._action_tokenizer(actions_dummy[0]) + action_string

        if not language_prompt:
            convo_text = "No language instruction"
        else:
            convo_text = f"What action should the robot take to {language_prompt}?"
        prompt_builder = self._prompt_builder_cls("openvla")
        prompt_builder.add_turn("human", convo_text)
        prompt_builder.add_turn("gpt", action_string)

        base_tok = self._processor.tokenizer
        input_ids = torch.tensor(
            base_tok(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        )
        labels = input_ids.clone()
        labels[: -(len(action_string) + 1)] = IGNORE_INDEX

        img_xform = self._processor.image_processor.apply_transform
        pixel_values_cur = img_xform(cur224)
        pixel_values_goal = img_xform(goal224)
        pixel_values = torch.cat(
            (pixel_values_cur.unsqueeze(0), pixel_values_goal.unsqueeze(0)), dim=1
        )  # [1, 2C, H, W]

        attention_mask = input_ids.ne(base_tok.pad_token_id).unsqueeze(0)
        input_ids = input_ids.unsqueeze(0)
        labels = labels.unsqueeze(0)

        device = self._device
        modality_t = torch.as_tensor([modality_id], dtype=torch.float32).to(
            torch.bfloat16
        ).to(device)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = self._vla(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                pixel_values=pixel_values.to(torch.bfloat16).to(device),
                modality_id=modality_t,
                labels=labels.to(device),
                output_hidden_states=True,
                proprio=goal_pose_t.to(torch.bfloat16).to(device),
                proprio_projector=self._pose_projector,
                noisy_actions=None,
                noisy_action_projector=None,
                diffusion_timestep_embeddings=None,
                use_film=False,
            )

        # Extract action-token hidden states (mirrors run_omnivla.py).
        gt_token_ids = labels[:, 1:].to(device)
        cur_mask = self._get_current_action_mask(gt_token_ids)
        next_mask = self._get_next_actions_mask(gt_token_ids)
        last_hidden = output.hidden_states[-1]
        text_hidden = last_hidden[:, self._num_patches:-1]
        actions_hidden = (
            text_hidden[cur_mask | next_mask]
            .reshape(1, self._chunk_len * self._action_dim, -1)
            .to(torch.bfloat16)
        )

        predicted_actions = self._action_head.predict_action(
            actions_hidden, modality_t
        )
        return predicted_actions, modality_id


def _load_full_ckpt(module_name: str, path: str, step: int, device: torch.device) -> dict:
    """Load a `<module>--<step>_checkpoint.pt` file, with the
    `pose_projector → proprio_projector` filename fallback used in
    inference/run_omnivla.py."""
    file_name = f"{module_name}--{step}_checkpoint.pt"
    candidate = os.path.join(path, file_name)
    if not os.path.exists(candidate) and module_name == "pose_projector":
        candidate = os.path.join(path, f"proprio_projector--{step}_checkpoint.pt")
    if not os.path.exists(candidate):
        raise FileNotFoundError(f"missing checkpoint: {candidate}")
    state_dict = torch.load(candidate, map_location=device)
    # Strip DDP "module." prefix if it was saved that way.
    return {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}


# ---- Factory ----------------------------------------------------------------
def build_model(cfg: ModelConfig) -> OmniVlaModelProto:
    if cfg.name in EDGE_NAMES:
        return OmniVlaEdgeModel(cfg)
    if cfg.name in FULL_NAMES:
        return OmniVlaModel(cfg)
    raise ValueError(
        f"unknown model.name: {cfg.name!r}; expected one of: "
        f"{', '.join(VALID_NAMES)}"
    )
