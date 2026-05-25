# `materialize` pulls in dataset deps (dlimp, tf, etc.) that aren't needed
# for inference; tolerate failure so `constants` and `action_tokenizer` stay
# importable.
try:
    from .materialize import get_vla_dataset_and_collator
except ImportError:
    pass
