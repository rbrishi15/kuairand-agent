"""[Sarthak — PLACEHOLDER] Priority 5, Day-3 low-risk wrap-up.

Blend predictions from the top 2-3 checkpoints the loop already produced
(weighted average or rank-average). Every checkpoint is
{"state_dict", "config", "val_metrics"} regardless of model (CLAUDE.md §5)
— load each with torch.load(path, weights_only=False) (state dicts here
can hold numpy arrays, e.g. the FM baseline, not just torch tensors, so
the default weights_only=True restriction would reject them).
"""


def ensemble_predict(checkpoint_paths, X, weights=None):
    raise NotImplementedError('TODO(Sarthak): load checkpoints, blend scores')
