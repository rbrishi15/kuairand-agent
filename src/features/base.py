"""[Min — PLACEHOLDER] ID + crossed features (FM-style), feeding DeepFMMTL.

CLAUDE.md §8 Priority 1. The playbook's already-tested findings (don't
repeat these): adding CWM's extra 13 feature domains gave no measurable
lift over the current 5 fields, and neither did bumping embedding dim
(k=8/16/32) — the user_id x video_id cross already absorbs most of the
learnable signal. Headroom is more likely in the loss function or
behavioral sequences (§8 Priorities 1-2) than in more static features.
"""


def build_base_features(splits):
    raise NotImplementedError('TODO(Min): produce the crossed feature tensor DeepFMMTL consumes')
