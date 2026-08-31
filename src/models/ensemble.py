"""[Rishi, implementing on Sarthak's behalf per explicit direction — his
file/priority, see CLAUDE.md §1] Priority 5: blend predictions from
existing checkpoints.

Every checkpoint is {"state_dict", "config", "val_metrics"} regardless of
model (CLAUDE.md §5) — load each with torch.load(path,
weights_only=False) (state dicts here can hold numpy arrays, e.g. the FM
baseline, not just torch tensors, so the default weights_only=True
restriction would reject them).

Blends via **rank-average**, not raw-score-average: FM's raw logit and
DeepFMMTL's raw logit aren't on comparable scales, and only within-user
*ordering* is ever scored (CLAUDE.md's within-user ranking task) — so each
checkpoint's score is converted to a percentile rank within each user's own
row group before blending, sidestepping the scale-mismatch entirely rather
than needing calibration.
"""
import numpy as np
import torch

from src.scoring import score


def _percentile_rank_within_user(scores, users):
    """[0, 1] percentile rank of each score within its own user's group.
    A lone-row user gets 0.5 (no ordering information to rank against).
    Plain argsort-of-argsort, not tie-averaged — ties are rare with
    continuous model outputs, and exact tie handling doesn't matter for a
    blend input the way it does for evaluate.py's own AUC computation.
    """
    scores = np.asarray(scores, dtype=np.float64)
    users = np.asarray(users)
    order = np.argsort(users, kind='stable')
    sorted_users = users[order]
    sorted_scores = scores[order]
    _, start_idx, counts = np.unique(sorted_users, return_index=True, return_counts=True)

    ranks = np.empty(len(scores), dtype=np.float64)
    for start, count in zip(start_idx, counts):
        if count == 1:
            ranks[start] = 0.5
        else:
            group = sorted_scores[start:start + count]
            ranks[start:start + count] = np.argsort(np.argsort(group, kind='stable')) / (count - 1)

    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    return ranks[inv]


def ensemble_predict(checkpoint_paths, X, dim, users, splits=None, split_name=None, weights=None):
    """Rank-average ensemble across arbitrary checkpoints — any mix of
    model types, any mix of techniques.

    checkpoint_paths: paths to blend.
    X, users: the (encoded features, user ids) for the split being scored
        — same shape/order as enc[split_name] — must come from encode()
        against the SAME dataset/config every checkpoint here was itself
        trained on, since `dim` (the shared embedding-table size) is
        passed once for all of them, not re-derived per checkpoint.
    splits, split_name: passed through to score() for any seq-encoder
        checkpoint that needs to rebuild sequence embeddings.
    weights: per-checkpoint blend weight, defaults to uniform. Pass
        unequal weights to weight by *technique* rather than by checkpoint
        count — e.g. 5 base-seed + 2 seq-seed checkpoints, each side
        wanting equal say, is weights=[0.1]*5 + [0.25]*2 (0.5 total each
        side), not naive uniform (which would let the group with more
        seeds dominate simply by outnumbering the other).
    """
    n = len(checkpoint_paths)
    if n == 0:
        raise ValueError('checkpoint_paths must not be empty')
    weights = weights if weights is not None else [1.0 / n] * n
    if len(weights) != n:
        raise ValueError(f'weights length ({len(weights)}) must match checkpoint_paths length ({n})')

    blended = np.zeros(len(X), dtype=np.float64)
    for path, w in zip(checkpoint_paths, weights):
        ckpt = torch.load(path, weights_only=False)
        raw = score(ckpt['config'], ckpt['state_dict'], X, dim, splits=splits, split_name=split_name)
        blended += w * _percentile_rank_within_user(raw, users)
    return blended
