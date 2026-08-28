"""[Nandit — PLACEHOLDER] Priority 4, opportunistic Day-3 only.

Masked-item prediction over interaction sequences, used only to pretrain
SeqEncoder's embeddings before supervised fine-tuning. CLAUDE.md §8: labels
aren't scarce here so expected lift is modest — run as one logged
experiment, keep only if it beats the multi-task model without it. First
thing to cut if time is short.
"""


def pretrain_masked_item(sequences, seq_encoder):
    raise NotImplementedError('TODO(Nandit, opportunistic): masked-item pretext task')
