# Top-level re-exports rely on the full training stack (FSDP, wandb, datasets);
# tolerate import failure so inference-only environments (e.g. NVIDIA's Jetson
# torch wheel, which omits torch.distributed) can still `import prismatic`.
try:
    from .models import available_model_names, available_models, get_model_description, load
except ImportError:
    pass
