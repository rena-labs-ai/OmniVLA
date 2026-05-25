"""OmniVLA model wrappers.

Two model variants share the same `forward()` signature so the handler doesn't
need to know which one it has:
  * `OmniVlaEdgeModel` — the small EfficientNet-b0 + transformer head model
    (`omnivla-edge`). Implemented today.
  * `OmniVlaModel`     — the full OpenVLA-OFT-based model (`omnivla` /
    `omnivla-finetuned-cast`). Stub for now; raises NotImplementedError until
    the full loader is ported from `inference/run_omnivla.py`.

Pick between them via `model.mode` in `config.yaml`.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch

# The inference/ helpers (`load_model`, `transform_*`) are imported as flat
# modules — add them to sys.path the same way `run_omnivla_edge.py` does.
_OMNIVLA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_OMNIVLA_ROOT, os.path.join(_OMNIVLA_ROOT, "inference")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import clip  # noqa: E402
from utils_policy import load_model  # noqa: E402


# Each name doubles as the HF repo (NHirose/<name>) and the directory name
# under the OmniVLA repo root. Adding a new checkpoint = one entry here.
EDGE_NAMES = ("omnivla-edge",)
FULL_NAMES = ("omnivla-original", "omnivla-original-balance", "omnivla-finetuned-cast")
VALID_NAMES = EDGE_NAMES + FULL_NAMES


# ---- Config ------------------------------------------------------------------
@dataclass
class ModelConfig:
    """The only knob is the checkpoint name; everything else is derived."""

    name: str
    device: str = "cuda:0"


# Fixed architecture params for the OmniVLA-edge checkpoint. These are tied to
# the trained weights, not user-tunable, so they live next to the loader.
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


def _weights_path_for(name: str) -> str:
    """Resolve `<OmniVLA-root>/<name>/<name>.pth` for edge checkpoints, or
    `<OmniVLA-root>/<name>/` for full-OmniVLA HF repos."""
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
    def tokenize(self, text: str) -> torch.Tensor: ...
    def forward(
        self,
        obs_images: torch.Tensor,
        cur_large: torch.Tensor,
        goal_pose: torch.Tensor,
        goal_image: torch.Tensor,
        map_images: torch.Tensor,
        language_tokens: torch.Tensor,
        modality_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


# ---- OmniVLA-edge wrapper (implemented) --------------------------------------
class OmniVlaEdgeModel:
    """Loaded OmniVLA-edge model + CLIP text encoder."""

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
        map_images: torch.Tensor,        # [B, 9, 352, 352]
        language_tokens: torch.Tensor,
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


# ---- Full OmniVLA wrapper (stub) ---------------------------------------------
class OmniVlaModel:
    """Loaded full OmniVLA model (OpenVLA-OFT family) + CLIP text encoder.

    Not yet wired. The full model needs the prismatic + transformers fork +
    flash-attn deps and a different preprocessing pipeline; see
    `inference/run_omnivla.py` for the reference implementation that has to be
    ported into this wrapper. Until then, instantiating this class raises so
    you fail loud at config-load time instead of silently running the wrong
    model.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        # Fail loud on missing weights before the NotImplementedError below —
        # gives the user the exact HF clone command instead of a stub message.
        path = _weights_path_for(cfg.name)
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"OmniVLA weights directory not found: {path}\n"
                f"Download with:  git lfs install && "
                f"git clone https://huggingface.co/NHirose/{cfg.name} {path}"
            )
        raise NotImplementedError(
            "Full OmniVLA model is not wired yet. Set `model.name: omnivla-edge` "
            "in config.yaml, or port the loader from `inference/run_omnivla.py` "
            "into OmniVlaModel.__init__ / forward()."
        )

    @property
    def device(self) -> torch.device:
        raise NotImplementedError

    def tokenize(self, text: str) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


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
