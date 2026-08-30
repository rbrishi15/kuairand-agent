"""KuaiRand-Pure baselines。
  --model pop   : item popularity（官方 baseline，纯统计，不训练）
  --model fm    : Factorization Machine（起步模型，学生从这里往上改）
  --model random: 随机打分（下界，用来自检评测代码没坏）
只依赖 numpy。用法见 README.md
"""
import argparse, collections, time
import numpy as np
from data import load, encode, FIELDS
from evaluate import evaluate

def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

# ---------------- item popularity（官方 baseline） ----------------
def run_pop(splits, prior=20.0):
    pos, imp = collections.Counter(), collections.Counter()
    for x in splits['train']:
        imp[x[2]] += 1; pos[x[2]] += x[6]
    gmean = sum(pos.values()) / sum(imp.values())
    score = lambda v: (pos[v] + prior * gmean) / (imp[v] + prior) if imp[v] else gmean
    out = {}
    for name in ('valid', 'test'):
        rws = splits[name]
        out[name] = evaluate([x[1] for x in rws], [x[6] for x in rws],
                             [score(x[2]) for x in rws])
    return out

def run_random(splits, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for name in ('valid', 'test'):
        rws = splits[name]
        out[name] = evaluate([x[1] for x in rws], [x[6] for x in rws],
                             rng.random(len(rws)))
    return out

# ---------------- Factorization Machine ----------------
class FM:
    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed=0, init_V=None, frozen_rows=None):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        if init_V is not None:
            assert init_V.shape == self.V.shape, f"init_V shape {init_V.shape} != {self.V.shape}"
            self.V = init_V.astype(np.float32).copy()
        # frozen_rows: 1D array of row indices into V that should never be updated
        # (no gradient, no L2 decay, no Adam moment update) -- protects SSL-pretrained
        # rows for rarely-seen items from dense L2 decay + stale Adam second-moment
        # estimates during sparse updates.
        self.frozen = None
        if frozen_rows is not None:
            self.frozen = np.zeros(dim, dtype=bool)
            self.frozen[frozen_rows] = True
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr, self.l2 = lr, l2
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.t = 0

    def logits(self, X):
        E = self.V[X]                                   # (B,F,k)
        S = E.sum(1)                                    # (B,k)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + inter, E, S

    def _update(self, X, E, S, dz):
        """One Adam step on V/W/b from per-row dL/dz (already batch-averaged).
        Factored out of step() so the pairwise / listwise objectives below reuse
        the exact same parameter-gradient math and optimizer state -- the only
        thing that changes between the three losses is how dz is computed."""
        dz = np.ascontiguousarray(dz, dtype=np.float32)
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        np.add.at(gW, X, dz[:, None])
        np.add.at(gV, X, dz[:, None, None] * (S[:, None, :] - E))
        gV += self.l2 * self.V; gW += self.l2 * self.W
        if self.frozen is not None:
            gV[self.frozen] = 0.0
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        self.b -= self.lr * float(dz.sum())

    def step(self, X, y):
        """Pointwise logloss -- the official baseline objective. Unchanged."""
        B = len(y)
        z, E, S = self.logits(X)
        p = sigmoid(z)
        self._update(X, E, S, (p - y) / B)
        return float(-np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)))

    def step_bpr(self, Xpos, Xneg):
        """Pairwise BPR. Xpos[i] and Xneg[i] are a positive/negative impression
        drawn from the SAME user; maximize sigmoid(z_pos - z_neg). One forward
        over the stacked rows, then _update() with the pairwise dL/dz.
        dL/dz_pos = -sigmoid(z_neg - z_pos)/n ,  dL/dz_neg = +sigmoid(...)/n .
        (Bias cancels: sum(dz) == 0, and a per-user constant never changes the
        within-user order anyway.)"""
        n = len(Xpos)
        Xall = np.concatenate([Xpos, Xneg])
        z, E, S = self.logits(Xall)
        zp, zn = z[:n], z[n:]
        wrong = sigmoid(zn - zp)                       # P(model orders the pair wrong)
        dz = np.empty(2 * n, dtype=np.float32)
        dz[:n] = -wrong / n
        dz[n:] = wrong / n
        self._update(Xall, E, S, dz)
        return float(-np.mean(np.log(sigmoid(zp - zn) + 1e-9)))

    def step_list(self, X, y, group_sizes):
        """Listwise softmax cross-entropy within each user's impression list:
        P = softmax(scores) over that user's rows, target Q = uniform over the
        user's positives, loss = CE(Q, P), dL/dz = P - Q. Users with no positive
        row carry no gradient (nothing to rank), mirroring GAUC's exclusion of
        all-same-label users. Rows of X are user-contiguous; group_sizes gives
        each user's row count and sums to len(X)."""
        z, E, S = self.logits(X)
        starts = np.concatenate([[0], np.cumsum(group_sizes)[:-1]]).astype(np.int64)
        seg = np.repeat(np.arange(len(group_sizes)), group_sizes)
        ez = np.exp(z - np.maximum.reduceat(z, starts)[seg])
        P = ez / np.add.reduceat(ez, starts)[seg]
        possum = np.add.reduceat(y, starts)
        valid_row = possum[seg] > 0
        nv = int((possum > 0).sum())
        if nv == 0:
            return 0.0
        Q = y / np.where(possum[seg] > 0, possum[seg], 1.0)
        dz = ((P - Q) / nv).astype(np.float32)
        dz[~valid_row] = 0.0
        self._update(X, E, S, dz)
        return float(-np.sum((Q * np.log(P + 1e-9))[valid_row]) / nv)

    def predict(self, X, bs=200_000):
        return np.concatenate([self.logits(X[i:i + bs])[0] for i in range(0, len(X), bs)])

def _user_groups(users):
    """user_id -> int64 array of its row indices, in file order."""
    g = collections.defaultdict(list)
    for i, u in enumerate(users):
        g[u].append(i)
    return {u: np.asarray(v, dtype=np.int64) for u, v in g.items()}


def _epoch_bpr(m, X, y, groups, rng, bs, max_pairs_per_user):
    """Sample (positive, negative) row pairs per user, shuffle, minibatch."""
    pos_l, neg_l = [], []
    for rows in groups.values():
        yl = y[rows]
        pos, neg = rows[yl == 1], rows[yl == 0]
        if len(pos) == 0 or len(neg) == 0:
            continue                      # no gradient available for this user
        npair = min(max_pairs_per_user, max(len(pos), len(neg)))
        pos_l.append(rng.choice(pos, size=npair))
        neg_l.append(rng.choice(neg, size=npair))
    if not pos_l:
        return [0.0]
    pi = np.concatenate(pos_l); nj = np.concatenate(neg_l)
    perm = rng.permutation(len(pi)); pi, nj = pi[perm], nj[perm]
    return [m.step_bpr(X[pi[s:s + bs]], X[nj[s:s + bs]]) for s in range(0, len(pi), bs)]


def _epoch_listwise(m, X, y, groups, rng, users_per_batch):
    """Minibatch users; each step ranks every impression of the users in the batch."""
    us = list(groups.keys()); rng.shuffle(us)
    out = []
    for s in range(0, len(us), users_per_batch):
        chunk = us[s:s + users_per_batch]
        rows = np.concatenate([groups[u] for u in chunk])
        sizes = np.fromiter((len(groups[u]) for u in chunk), dtype=np.int64, count=len(chunk))
        out.append(m.step_list(X[rows], y[rows], sizes))
    return out


def run_fm(splits, k=16, lr=0.001, epochs=40, bs=8192, patience=4, seed=0, verbose=True,
           init_V=None, return_model=False, frozen_rows=None, loss='pointwise',
           max_pairs_per_user=100, users_per_batch=1024):
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    m = FM(dim, k=k, lr=lr, seed=seed, init_V=init_V, frozen_rows=frozen_rows)
    rng = np.random.default_rng(seed)
    groups = _user_groups(utr) if loss != 'pointwise' else None
    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        if loss == 'pointwise':
            idx = rng.permutation(len(ytr))
            losses = [m.step(Xtr[idx[i:i + bs]], ytr[idx[i:i + bs]]) for i in range(0, len(idx), bs)]
        elif loss == 'bpr':
            losses = _epoch_bpr(m, Xtr, ytr, groups, rng, bs, max_pairs_per_user)
        elif loss == 'listwise':
            losses = _epoch_listwise(m, Xtr, ytr, groups, rng, users_per_batch)
        else:
            raise ValueError(f"unknown loss {loss!r} (pointwise|bpr|listwise)")
        va = evaluate(uva, yva, m.predict(Xva))
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} "
                  f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= patience:
                if verbose: print(f"  early stop at epoch {ep}")
                break
        m.V, m.W, m.b = best_state
    out = {'valid': evaluate(uva, yva, m.predict(Xva)),
           'test':  evaluate(ute, yte, m.predict(Xte))}
    if return_model:
        out['model'] = m
        out['test_scores'] = m.predict(Xte)
        out['test_users'] = ute
        out['test_labels'] = yte
    return out
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data',
                    help='KuaiRand-Pure 解压后的 data 目录')
    ap.add_argument('--model', default='fm', choices=['pop', 'fm', 'random'])
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--loss', default='pointwise', choices=['pointwise', 'bpr', 'listwise'],
                    help='fm training objective: pointwise logloss (official baseline), '
                         'pairwise BPR, or listwise within-user softmax')
    a = ap.parse_args()
    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}")
    res = {'pop': run_pop, 'random': lambda s: run_random(s, a.seed),
           'fm': lambda s: run_fm(s, k=a.k, lr=a.lr, epochs=a.epochs, seed=a.seed, loss=a.loss)}[a.model](splits)
    print(f"\n=== {a.model}{'' if a.model != 'fm' else ' ['+a.loss+']'} (seed={a.seed}) ===")
    for sp in ('valid', 'test'):
        r = res[sp]
        print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
