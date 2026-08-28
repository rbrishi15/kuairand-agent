"""[Nandit — PLACEHOLDER] Priority 3: GRU/attention encoder over a user's
recent interaction history (from src/features/sequential.py).

Output plugs into DeepFMMTL.forward(x, seq_embedding=...) — that hook
already exists in src/models/deepfm_mtl.py.
"""
import torch.nn as nn


class SeqEncoder(nn.Module):
    def __init__(self, num_items, embed_dim=16, hidden_dim=32):
        super().__init__()
        raise NotImplementedError('TODO(Nandit): GRU or single-layer attention over history')

    def forward(self, seq_ids):
        raise NotImplementedError('TODO(Nandit): return a per-user embedding')
