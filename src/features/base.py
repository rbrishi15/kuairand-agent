"""[Min] ID + crossed features (FM-style), feeding DeepFMMTL.

CLAUDE.md §8 Priority 1. Already-tested findings (don't repeat these):
adding CWM's extra 13 feature domains gave no measurable lift over the
current 5 fields, and neither did bumping embedding dim (k=8/16/32) — the
user_id x video_id cross already absorbs most of the learnable signal.
Headroom is more likely in the loss function or behavioral sequences
(§8 Priorities 1-2) than in more static features.

Deliberately a thin wrapper, not a re-implementation: src.data.encode() is
the single source of truth for vocab/id derivation (CLAUDE.md §5) and is
already FM-style (global ids into one shared embedding table, offsets
baked in) — DeepFMMTL consumes exactly that shape. This seam exists so a
future DeepFM-specific feature (e.g. an explicit wide-side cross not worth
adding to the shared encoder) has somewhere to live without touching
src/data.py, not to duplicate what's already there.
"""
from src.data import encode


def build_base_features(splits):
    """splits: the dict returned by src.data.load(config) (raw rows, not
    yet encoded). Returns (enc, dim) exactly as src.data.encode() does:
    enc[name] = (X, y, users) per split, dim = total embedding-table size.
    """
    return encode(splits)
