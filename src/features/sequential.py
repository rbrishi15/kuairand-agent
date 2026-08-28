"""[Nandit — PLACEHOLDER] Priority 3: per-user recent-interaction sequences.

Feeds src/models/seq_encoder.py. CLAUDE.md §8: current features use zero
behavioral history despite each user having hundreds to thousands of
interactions in train — this is a completely unexplored direction.
"""


def build_sequences(splits, max_len=50):
    raise NotImplementedError('TODO(Nandit): per-user ordered interaction history')
