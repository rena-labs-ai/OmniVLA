"""OmniVLA edge model wrapper.

Loads the checkpoint once and exposes a single `forward()` that takes
already-preprocessed tensors. Keeps all knowledge of `prismatic`,
`OmniVLA_edge`, and the CLIP text encoder confined to this file; the
`Handler` only sees tensors in and waypoints out, so it can be unit-tested
with a fake `OmniVlaModel`.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any

import torch

# The inference/ helpers (`load_model`, `transform_*`) are imported as flat
# modules — add them to sys.path the same way `run_omnivla_edge.py` does.
_OMNIVLA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_OMNIVLA_ROOT, os.path.join(_OMNIVLA_ROOT, "inference")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import clip  # noqa: E402
from utils_policy import load_model  # noqa: E402


@dataclass
class ModelConfig:
    """Hyperparameters for the OmniVLA-edge checkpoint."""

    weights_path: str
    device: str = "cuda:0"
    model_type: str = "omnivla-edge"
    len_traj_pred: int = 8
    context_size: int = 5
    learn_angle: bool = True
    obs_encoder: str = "efficientnet-b0"
    encoding_size: int = 256
    obs_encoding_size: int = 1024
    goal_encoding_size: int = 1024
    late_fusion: bool = False
    mha_num_attention_heads: int = 4
    mha_num_attention_layers: int = 4
    mha_ff_dim_factor: int = 4
    clip_type: str = "ViT-B/32"

    def to_params(self) -> dict[str, Any]:
        # `load_model` consumes a plain dict; this keeps the field names aligned
        # with the OmniVLA-edge config schema.
        return {
            "model_type": self.model_type,
            "len_traj_pred": self.len_traj_pred,
            "context_size": self.context_size,
            "learn_angle": self.learn_angle,
            "obs_encoder": self.obs_encoder,
            "encoding_size": self.encoding_size,
            "obs_encoding_size": self.obs_encoding_size,
            "goal_encoding_size": self.goal_encoding_size,
            "late_fusion": self.late_fusion,
            "mha_num_attention_heads": self.mha_num_attention_heads,
            "mha_num_attention_layers": self.mha_num_attention_layers,
            "mha_ff_dim_factor": self.mha_ff_dim_factor,
            "clip_type": self.clip_type,
        }


class OmniVlaModel:
    """Loaded OmniVLA-edge model + CLIP text encoder."""

    def __init__(self, cfg: ModelConfig) -> None:
        if not os.path.exists(cfg.weights_path):
            raise FileNotFoundError(f"weights not found: {cfg.weights_path}")
        self._device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        model, text_encoder, _preprocess = load_model(
            cfg.weights_path, cfg.to_params(), self._device
        )
        self._model = model.to(self._device).eval()
        self._text_encoder = text_encoder.to(self._device).eval()

    @property
    def device(self) -> torch.device:
        return self._device

    def tokenize(self, text: str) -> torch.Tensor:
        # CLIP tokenizer truncates; empty / placeholder text is handled by the
        # caller passing modality_id != language modes.
        return clip.tokenize(text or "xxxx", truncate=True).to(self._device)

    @torch.no_grad()
    def forward(
        self,
        obs_images: torch.Tensor,        # [B, 3*context, 96, 96]
        cur_large: torch.Tensor,         # [B, 3, 224, 224]
        goal_pose: torch.Tensor,         # [B, 4]
        goal_image: torch.Tensor,        # [B, 3, 96, 96]
        map_images: torch.Tensor,        # [B, 9, 352, 352] (sat_cur, sat_goal, obs_cur)
        language_tokens: torch.Tensor,   # CLIP token tensor
        modality_id: torch.Tensor,       # [B]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run inference. Returns (predicted_actions, modality_id_used)."""
        feat_text = self._text_encoder.encode_text(language_tokens)
        bimg = goal_image.size(0)
        predicted_actions, _distances, mask_number = self._model(
            obs_images.repeat(bimg, 1, 1, 1),
            goal_pose.repeat(bimg, 1),
            map_images.repeat(bimg, 1, 1, 1),
            goal_image,
            modality_id.repeat(bimg),
            feat_text.repeat(bimg, 1),
            cur_large.repeat(bimg, 1, 1, 1),
        )
        return predicted_actions, mask_number
