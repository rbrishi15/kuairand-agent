"""[Min] Priority 1: multi-task DeepFM — primary model (CLAUDE.md §8).

Primary head predicts `long_view`; auxiliary heads exist for `click`,
`like`, `play_time` but aren't wired into src/train.py's training path yet
— src/data.py's load()/encode() (Rishi's file) only exposes `long_view`
today, no is_click/is_like/play_time_ms. Flagged rather than silently
extended (CLAUDE.md §0.2/§0.4): once those labels exist, pass them as
`aux_targets` to compute_loss() below, no architecture change needed.

field_dims is the TOTAL embedding-table size — i.e. the `dim` returned by
src.data.encode(), not a per-field cardinality list. `x` is expected
pre-offset exactly as encode() produces it (global ids into one shared
table), matching src/models/fm.py's convention so DeepFMMTL is a drop-in
alternative to FM via train.py's MODEL_REGISTRY. num_fields must equal
X.shape[1] (5, per src.data.FIELDS, unless src/features/base.py's caller
passes something else).

forward()'s two optional kwargs are the load-bearing interface contract
(CLAUDE.md §5) and must never be removed:
  - seq_embedding: Nandit's per-user sequence embedding (B, seq_dim),
    concatenated into the deep tower's input when provided. Construct the
    model with seq_dim > 0 to size the deep tower for it up front.
  - sample_weight: Vidush's IPS weights (B,). Reweighting is a loss-time
    concern, not a forward-pass one, so forward() just threads it through
    to compute_loss() rather than consuming it here.

Checkpoint contract (CLAUDE.md §5): {"state_dict": ..., "config": ...,
"val_metrics": {...}} via torch.save — nn.Module.state_dict() already
satisfies the state_dict half, no wrapping needed.
"""
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.evaluate import evaluate


class DeepFMMTL(nn.Module):
    def __init__(self, field_dims, embed_dim=16, num_fields=5, aux_tasks=(),
                 hidden_dims=(64, 32), dropout=0.1, seq_dim=0):
        super().__init__()
        self.num_fields = num_fields
        self.aux_tasks = tuple(aux_tasks)

        self.embed = nn.Embedding(field_dims, embed_dim)
        self.linear = nn.Embedding(field_dims, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.embed.weight, 0, 0.01)
        nn.init.zeros_(self.linear.weight)

        prev = num_fields * embed_dim + seq_dim
        layers = []
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.deep = nn.Sequential(*layers)
        self.deep_out = nn.Linear(prev, 1)
        self.aux_heads = nn.ModuleDict({t: nn.Linear(prev, 1) for t in self.aux_tasks})

    def forward(self, x, sample_weight=None, seq_embedding=None):
        e = self.embed(x)                                          # (B, F, k)
        s = e.sum(1)
        fm_term = 0.5 * ((s ** 2).sum(1) - (e ** 2).sum((1, 2)))    # (B,)
        linear_term = self.linear(x).squeeze(-1).sum(1)             # (B,)

        deep_in = e.flatten(1)
        if seq_embedding is not None:
            deep_in = torch.cat([deep_in, seq_embedding], dim=1)
        h = self.deep(deep_in)

        primary = self.bias + linear_term + fm_term + self.deep_out(h).squeeze(-1)
        aux = {name: head(h).squeeze(-1) for name, head in self.aux_heads.items()}
        return {'primary': primary, 'aux': aux, 'sample_weight': sample_weight}

    def compute_loss(self, out, y_primary, aux_targets=None, aux_weight=0.3):
        """BCE on the primary head (IPS-reweighted if sample_weight is
        present), plus aux_weight * mean BCE over any aux heads that have a
        matching entry in aux_targets — heads without one are skipped, so
        aux_tasks can exist structurally before real labels are wired in.
        """
        w = out['sample_weight']

        def bce(logits, target):
            if w is None:
                return F.binary_cross_entropy_with_logits(logits, target)
            per_ex = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
            return (per_ex * w).sum() / w.sum()

        total = bce(out['primary'], y_primary)
        aux_targets = aux_targets or {}
        aux_losses = [bce(out['aux'][name], target)
                      for name, target in aux_targets.items() if name in out['aux']]
        if aux_losses:
            total = total + aux_weight * torch.stack(aux_losses).mean()
        return total


def run_deepfm_mtl(enc, dim, embed_dim=16, num_fields=5, lr=0.001, epochs=40,
                    bs=8192, patience=4, seed=0, verbose=True):
    """Mirrors src/models/fm.py's run_fm() shape so train.py can dispatch to
    either symmetrically. CPU by default (CLAUDE.md §3: every model needs a
    CPU code path); uses CUDA only if available, never requires it.
    Primary-only for now — see module docstring for why aux heads aren't
    fed real targets yet.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed)

    Xtr, ytr, _ = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    Xtr_t = torch.from_numpy(Xtr).long().to(device)
    ytr_t = torch.from_numpy(ytr).float().to(device)
    Xva_t = torch.from_numpy(Xva).long().to(device)
    Xte_t = torch.from_numpy(Xte).long().to(device)

    model = DeepFMMTL(dim, embed_dim=embed_dim, num_fields=num_fields).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def predict(X_t, bs=200_000):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X_t), bs):
                out.append(model(X_t[i:i + bs])['primary'].cpu().numpy())
        return np.concatenate(out)

    best, best_state, bad = -1, None, 0
    n = len(ytr)
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        idx = torch.randperm(n, device=device)
        losses = []
        for i in range(0, n, bs):
            b = idx[i:i + bs]
            out = model(Xtr_t[b])
            loss = model.compute_loss(out, ytr_t[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        va = evaluate(uva, yva, predict(Xva_t))
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} "
                  f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"  early stop at epoch {ep}")
                break

    model.load_state_dict(best_state)
    return model, {'valid': evaluate(uva, yva, predict(Xva_t)),
                   'test': evaluate(ute, yte, predict(Xte_t))}
