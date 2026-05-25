# `load`/`materialize` pull in training-only deps; tolerate failure so the
# inference subpackages (extern.hf, backbones.llm.prompting, ...) remain
# importable on machines missing torch.distributed / wandb / dlimp.
try:
    from .load import available_model_names, available_models, get_model_description, load, load_vla
    from .materialize import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform, get_vlm
except ImportError:
    pass
