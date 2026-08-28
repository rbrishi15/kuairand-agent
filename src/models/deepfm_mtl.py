"""[Min — PLACEHOLDER, not an implementation] Priority 1: multi-task DeepFM.

Primary head predicts `long_view`; auxiliary heads can cover `click`,
`like`, `play_time` (CLAUDE.md §8, Priority 1).

The forward() signature below is the single most important interface
contract in the whole project (CLAUDE.md §5, playbook §5): both optional
kwargs must exist from day 1, even before Vidush's/Nandit's code is ready,
or they sit idle waiting on a rewrite once you add them later.
  - sample_weight: Vidush's IPS weights plug in here.
  - seq_embedding: Nandit's sequence encoder output plugs in here.
If you change this signature, tell Vidush and Nandit in the same message
as the commit.

Checkpoint contract (CLAUDE.md §5): save as
{"state_dict": self.state_dict(), "config": config, "val_metrics": {...}}
via torch.save — same shape every model uses, so Sarthak's ensembler needs
no per-model special-casing.
"""
import torch.nn as nn


class DeepFMMTL(nn.Module):
    def __init__(self, field_dims, embed_dim=16):
        super().__init__()
        raise NotImplementedError('TODO(Min): build the shared embedding + FM + deep tower')

    def forward(self, x, sample_weight=None, seq_embedding=None):
        raise NotImplementedError('TODO(Min): primary long_view head + auxiliary heads')
