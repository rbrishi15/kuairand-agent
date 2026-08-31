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
                    bs=8192, patience=4, seed=0, verbose=True,
                    sample_weight=None, seq_arrays=None, seq_kwargs=None):
    """Mirrors src/models/fm.py's run_fm() shape so train.py can dispatch to
    either symmetrically. CPU by default (CLAUDE.md §3: every model needs a
    CPU code path); uses CUDA only if available, never requires it.
    Primary-only for now — see module docstring for why aux heads aren't
    fed real targets yet.

    [Rishi — integration wiring, CLAUDE.md Definition of Done: "Min's
    sample_weight/seq_embedding hooks are actually used by Vidush's and
    Nandit's code, not sitting unused"] Two optional args close that loop:

      sample_weight: Vidush's IPS train-row weights
        (src.features.propensity.estimate_propensity(...)['train']),
        row-aligned to enc['train']. Threaded into DeepFMMTL.forward()'s
        existing sample_weight kwarg every training step; valid/test are
        always unweighted (evaluation must stay on the true distribution).
      seq_arrays: {'train','valid','test': int64 (N, max_len) arrays} from
        src.features.sequential.sequences_for_rows, row-aligned to enc[name].
        When given, a SeqEncoder (src.models.seq_encoder) is constructed
        from seq_kwargs (must include num_items) and trained *jointly* with
        DeepFMMTL — its output feeds DeepFMMTL.forward()'s seq_embedding
        kwarg every step, both forward and backward. Left on the returned
        model as `.seq_encoder` (None if unused) so callers can save/load
        its weights alongside DeepFMMTL's.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed)

    Xtr, ytr, _ = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    Xtr_t = torch.from_numpy(Xtr).long().to(device)
    ytr_t = torch.from_numpy(ytr).float().to(device)
    Xva_t = torch.from_numpy(Xva).long().to(device)
    Xte_t = torch.from_numpy(Xte).long().to(device)

    w_t = None
    if sample_weight is not None:
        w_t = torch.from_numpy(np.asarray(sample_weight, dtype=np.float32)).to(device)

    seq_encoder = None
    Str_t = Sva_t = Ste_t = None
    seq_dim = 0
    if seq_arrays is not None:
        from src.models.seq_encoder import SeqEncoder
        if not seq_kwargs or 'num_items' not in seq_kwargs:
            raise ValueError('seq_kwargs must include num_items when seq_arrays is given')
        seq_encoder = SeqEncoder(**seq_kwargs).to(device)
        seq_dim = seq_encoder.out_dim
        Str_t = torch.from_numpy(seq_arrays['train']).to(device)
        Sva_t = torch.from_numpy(seq_arrays['valid']).to(device)
        Ste_t = torch.from_numpy(seq_arrays['test']).to(device)

    model = DeepFMMTL(dim, embed_dim=embed_dim, num_fields=num_fields, seq_dim=seq_dim).to(device)
    params = list(model.parameters()) + (list(seq_encoder.parameters()) if seq_encoder else [])
    opt = torch.optim.Adam(params, lr=lr)

    def _set_mode(training):
        model.train(training)
        if seq_encoder is not None:
            seq_encoder.train(training)

    def predict(X_t, S_t=None, bs=200_000):
        _set_mode(False)
        out = []
        with torch.no_grad():
            for i in range(0, len(X_t), bs):
                se = seq_encoder(S_t[i:i + bs]) if seq_encoder is not None else None
                out.append(model(X_t[i:i + bs], seq_embedding=se)['primary'].cpu().numpy())
        return np.concatenate(out)

    best, best_state, best_seq_state, bad = -1, None, None, 0
    n = len(ytr)
    for ep in range(1, epochs + 1):
        _set_mode(True)
        t0 = time.time()
        idx = torch.randperm(n, device=device)
        losses = []
        for i in range(0, n, bs):
            b = idx[i:i + bs]
            se = seq_encoder(Str_t[b]) if seq_encoder is not None else None
            wb = w_t[b] if w_t is not None else None
            out = model(Xtr_t[b], sample_weight=wb, seq_embedding=se)
            loss = model.compute_loss(out, ytr_t[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        va = evaluate(uva, yva, predict(Xva_t, Sva_t))
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} "
                  f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if seq_encoder is not None:
                best_seq_state = {k: v.detach().clone() for k, v in seq_encoder.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"  early stop at epoch {ep}")
                break

    model.load_state_dict(best_state)
    if seq_encoder is not None:
        seq_encoder.load_state_dict(best_seq_state)
    # NOT `model.seq_encoder = seq_encoder`: nn.Module.__setattr__ treats any
    # nn.Module-valued attribute as a submodule to auto-register, which would
    # silently fold seq_encoder's params into model.state_dict() under a
    # "seq_encoder.*" prefix — then a fresh DeepFMMTL() (which never had
    # .seq_encoder set) rejects those as unexpected keys on load_state_dict.
    # object.__setattr__ bypasses that registration; this is a plain
    # attribute for the caller's convenience, not a submodule of model.
    object.__setattr__(model, 'seq_encoder', seq_encoder)
    return model, {'valid': evaluate(uva, yva, predict(Xva_t, Sva_t)),
                   'test': evaluate(ute, yte, predict(Xte_t, Ste_t))}


def _build_bpr_index(y, users):
    """Group train row indices by user into positive/negative arrays,
    keeping only users with at least one of each (mirrors evaluate()'s own
    GAUC exclusion rule: 0 < npos < len(user's rows) — a user with only one
    class can't form a pair either way).

    Returns (all_pos_idx, all_pos_code, neg_by_code):
      all_pos_idx  : int64 array, every eligible positive row's index
      all_pos_code : int64 array, same length, which eligible-user each
                     entry in all_pos_idx belongs to (0..num_eligible-1)
      neg_by_code  : {code: int64 array of that user's negative row indices}
    """
    from collections import defaultdict
    pos_by_user, neg_by_user = defaultdict(list), defaultdict(list)
    for i, (u, yy) in enumerate(zip(users, y)):
        (pos_by_user if yy > 0 else neg_by_user)[u].append(i)
    eligible = [u for u in pos_by_user if u in neg_by_user]
    if not eligible:
        raise ValueError('no train users have both a positive and a negative impression')

    all_pos_idx, all_pos_code, neg_by_code = [], [], {}
    for code, u in enumerate(eligible):
        all_pos_idx.extend(pos_by_user[u])
        all_pos_code.extend([code] * len(pos_by_user[u]))
        neg_by_code[code] = np.array(neg_by_user[u], dtype=np.int64)
    return (np.array(all_pos_idx, dtype=np.int64),
            np.array(all_pos_code, dtype=np.int64), neg_by_code)


def _sample_bpr_pairs(all_pos_idx, all_pos_code, neg_by_code, n_pairs, rng):
    """One epoch's (pos_idx, neg_idx) arrays: n_pairs positives sampled
    uniformly from all eligible positives, each paired with a fresh random
    negative from the *same* user. Vectorized per unique sampled user
    (not per pair) — with ~10-30k eligible users on KuaiRand-Pure this is
    a small Python loop regardless of how large n_pairs is.
    """
    sel = rng.integers(0, len(all_pos_idx), size=n_pairs)
    pos_idx = all_pos_idx[sel]
    codes = all_pos_code[sel]
    neg_idx = np.empty(n_pairs, dtype=np.int64)

    order = np.argsort(codes, kind='stable')
    sorted_codes = codes[order]
    unique_codes, start = np.unique(sorted_codes, return_index=True)
    start = np.append(start, n_pairs)
    for k in range(len(unique_codes)):
        lo, hi = start[k], start[k + 1]
        slots = order[lo:hi]
        negs = neg_by_code[unique_codes[k]]
        neg_idx[slots] = negs[rng.integers(0, len(negs), size=len(slots))]
    return pos_idx, neg_idx


def run_deepfm_mtl_bpr(enc, dim, embed_dim=16, num_fields=5, lr=0.001, epochs=40,
                        pairs_per_epoch=None, bs=8192, patience=4, seed=0, verbose=True,
                        seq_arrays=None, seq_kwargs=None):
    """BPR pairwise variant of run_deepfm_mtl (README "headroom" idea #1:
    pointwise BCE doesn't match GAUC/nDCG, which are ranking metrics). For
    each user with at least one positive AND one negative train impression,
    sample (pos, neg) pairs and train on -log(sigmoid(score_pos - score_neg))
    -- directly optimizes the same per-user pairwise ordering GAUC measures,
    instead of an independent per-row classification loss.

    Users with only one class contribute no pairs and are silently skipped
    (see _build_bpr_index) -- exactly the same exclusion GAUC itself applies.

    Scope, deliberately: no sample_weight support here. Pairwise IPS
    reweighting (reweight by which side of the pair? both? geometric mean?)
    is a real design question on its own, not attempted in this first cut --
    train.py raises rather than silently combining use_ips with loss=bpr.
    seq_embedding IS supported (same per-row lookup as the pointwise path,
    just applied to both the pos and neg batch each step).

    pairs_per_epoch defaults to the number of eligible positive rows -- each
    gets sampled ~once per epoch, paired with a fresh random negative from
    the same user.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    all_pos_idx, all_pos_code, neg_by_code = _build_bpr_index(ytr, utr)
    n_pairs = pairs_per_epoch or len(all_pos_idx)

    Xtr_t = torch.from_numpy(Xtr).long().to(device)
    Xva_t = torch.from_numpy(Xva).long().to(device)
    Xte_t = torch.from_numpy(Xte).long().to(device)

    seq_encoder = None
    Str_t = Sva_t = Ste_t = None
    seq_dim = 0
    if seq_arrays is not None:
        from src.models.seq_encoder import SeqEncoder
        if not seq_kwargs or 'num_items' not in seq_kwargs:
            raise ValueError('seq_kwargs must include num_items when seq_arrays is given')
        seq_encoder = SeqEncoder(**seq_kwargs).to(device)
        seq_dim = seq_encoder.out_dim
        Str_t = torch.from_numpy(seq_arrays['train']).to(device)
        Sva_t = torch.from_numpy(seq_arrays['valid']).to(device)
        Ste_t = torch.from_numpy(seq_arrays['test']).to(device)

    model = DeepFMMTL(dim, embed_dim=embed_dim, num_fields=num_fields, seq_dim=seq_dim).to(device)
    params = list(model.parameters()) + (list(seq_encoder.parameters()) if seq_encoder else [])
    opt = torch.optim.Adam(params, lr=lr)

    def _set_mode(training):
        model.train(training)
        if seq_encoder is not None:
            seq_encoder.train(training)

    def predict(X_t, S_t=None, bs=200_000):
        _set_mode(False)
        out = []
        with torch.no_grad():
            for i in range(0, len(X_t), bs):
                se = seq_encoder(S_t[i:i + bs]) if seq_encoder is not None else None
                out.append(model(X_t[i:i + bs], seq_embedding=se)['primary'].cpu().numpy())
        return np.concatenate(out)

    best, best_state, best_seq_state, bad = -1, None, None, 0
    for ep in range(1, epochs + 1):
        _set_mode(True)
        t0 = time.time()
        pos_idx, neg_idx = _sample_bpr_pairs(all_pos_idx, all_pos_code, neg_by_code, n_pairs, rng)
        order = rng.permutation(n_pairs)
        pos_idx, neg_idx = pos_idx[order], neg_idx[order]

        losses = []
        for i in range(0, n_pairs, bs):
            pb = torch.from_numpy(pos_idx[i:i + bs]).to(device)
            nb = torch.from_numpy(neg_idx[i:i + bs]).to(device)
            se_pos = seq_encoder(Str_t[pb]) if seq_encoder is not None else None
            se_neg = seq_encoder(Str_t[nb]) if seq_encoder is not None else None
            s_pos = model(Xtr_t[pb], seq_embedding=se_pos)['primary']
            s_neg = model(Xtr_t[nb], seq_embedding=se_neg)['primary']
            loss = -F.logsigmoid(s_pos - s_neg).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        va = evaluate(uva, yva, predict(Xva_t, Sva_t))
        if verbose:
            print(f"  epoch {ep:2d} | bpr_loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} "
                  f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if seq_encoder is not None:
                best_seq_state = {k: v.detach().clone() for k, v in seq_encoder.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"  early stop at epoch {ep}")
                break

    model.load_state_dict(best_state)
    if seq_encoder is not None:
        seq_encoder.load_state_dict(best_seq_state)
    object.__setattr__(model, 'seq_encoder', seq_encoder)  # see run_deepfm_mtl's comment above
    return model, {'valid': evaluate(uva, yva, predict(Xva_t, Sva_t)),
                   'test': evaluate(ute, yte, predict(Xte_t, Ste_t))}
