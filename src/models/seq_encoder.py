"""[Nandit] Priority 3: GRU encoder over a user's recent interaction history.

Consumes the padded id sequences from
``src.features.sequential.sequences_for_rows`` and produces one dense vector
per row, which plugs into Min's model via
``DeepFMMTL.forward(x, seq_embedding=<this>)`` (CLAUDE.md §5). Construct
``DeepFMMTL(..., seq_dim=SeqEncoder(...).out_dim)`` so the deep tower is
sized for the concatenated input.

CPU-only by construction (CLAUDE.md §3): a single-layer GRU, no custom CUDA,
runs anywhere PyTorch runs. GPU is a transparent speed-up if the caller
moves the module and inputs onto one.

The item-embedding table can be warm-started from
``src.models.ssl_pretrain.pretrain_item2vec`` output (Priority 4) by passing
``pretrained_emb`` — shapes must be ``(num_items + 1, embed_dim)``.
"""
import numpy as np
import torch
import torch.nn as nn


class SeqEncoder(nn.Module):
    def __init__(self, num_items, embed_dim=16, hidden_dim=32, pooling='gru',
                 pretrained_emb=None, freeze_emb=False):
        """``num_items``: size of the sequential video vocab (from
        :func:`src.features.sequential.build_video_vocab`). The table has
        ``num_items + 1`` rows; the last row (id ``num_items``) is PAD and is
        held at zero.

        ``pooling``: ``'gru'`` (default, Priority 3 proper) returns the final
        GRU hidden state; ``'mean'`` returns the mean of non-PAD item
        embeddings (cheap ablation baseline).
        """
        super().__init__()
        if pooling not in ('gru', 'mean'):
            raise ValueError(f"pooling must be 'gru' or 'mean', got {pooling!r}")
        self.num_items = num_items
        self.pad_idx = num_items
        self.pooling = pooling
        self.hidden_dim = hidden_dim

        self.item_emb = nn.Embedding(num_items + 1, embed_dim, padding_idx=self.pad_idx)
        nn.init.normal_(self.item_emb.weight, 0, 0.01)
        with torch.no_grad():
            self.item_emb.weight[self.pad_idx].zero_()
        if pretrained_emb is not None:
            self.load_pretrained(pretrained_emb, freeze=freeze_emb)

        if pooling == 'gru':
            self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
            self._out_dim = hidden_dim
        else:
            self.gru = None
            self._out_dim = embed_dim

    @property
    def out_dim(self):
        """Width of the vector produced by :meth:`forward` — pass this as
        ``seq_dim`` when constructing ``DeepFMMTL``."""
        return self._out_dim

    def load_pretrained(self, emb, freeze=False):
        """Copy an ``(num_items + 1, embed_dim)`` matrix (e.g. item2vec
        output) into the embedding table. The PAD row is re-zeroed regardless
        of what ``emb`` holds there."""
        w = torch.as_tensor(np.asarray(emb), dtype=torch.float32)
        if tuple(w.shape) != tuple(self.item_emb.weight.shape):
            raise ValueError(
                f"pretrained_emb shape {tuple(w.shape)} != "
                f"{tuple(self.item_emb.weight.shape)}")
        with torch.no_grad():
            self.item_emb.weight.copy_(w)
            self.item_emb.weight[self.pad_idx].zero_()
        self.item_emb.weight.requires_grad_(not freeze)

    def forward(self, seq_ids):
        """``seq_ids``: long tensor ``(B, L)`` of vocab ids, left-padded with
        ``self.pad_idx``. Returns ``(B, out_dim)``. An all-PAD row (cold-start
        user) returns a near-zero vector."""
        if seq_ids.dtype != torch.long:
            seq_ids = seq_ids.long()
        mask = seq_ids != self.pad_idx                      # (B, L) bool
        emb = self.item_emb(seq_ids)                        # (B, L, embed_dim)

        if self.pooling == 'mean':
            counts = mask.sum(1, keepdim=True).clamp(min=1)
            return (emb * mask.unsqueeze(-1)).sum(1) / counts

        lengths = mask.sum(1)                               # (B,)
        # Rows with length 0 would break pack_padded_sequence; run the GRU on
        # a clamped length and zero those outputs afterwards.
        safe_lengths = lengths.clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, safe_lengths, batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)                           # h_n: (1, B, hidden_dim)
        out = h_n.squeeze(0)                                # (B, hidden_dim)
        return out * (lengths > 0).unsqueeze(-1).to(out.dtype)
