# `materialize` (FSDPStrategy) and `metrics` (wandb) are training-only;
# tolerate failure so `train_utils` (used by inference) stays importable.
try:
    from .materialize import get_train_strategy
    from .metrics import Metrics, VLAMetrics
except ImportError:
    pass
