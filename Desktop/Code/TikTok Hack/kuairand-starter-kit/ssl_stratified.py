"""Stratify test rows by how often their video_id appeared in train (rare/medium/
popular tertiles), then compare random-init vs SSL warm-start FM separately in
each stratum. The aggregate primary score can hide a real effect if SSL helps
specifically on rare/cold-start videos but the majority of well-observed videos
already saturate under either initialization.

Prerequisite: run ssl_pretrain.py first to generate ssl_video_emb.npy.

Usage:
    python3 ssl_stratified.py [data_dir] [--ssl_emb ssl_video_emb.npy] [--seed 0]
"""
import argparse, collections
import numpy as np
from data import load, field_layout, FIELDS
from evaluate import evaluate
import baseline as B


def video_train_freq(splits):
    """video_id -> how many times it appeared in train."""
    c = collections.Counter(x[2] for x in splits['train'])
    return c


def stratify_indices(test_rows, freq):
    """Split test row indices into 3 roughly-equal-size buckets by train frequency
    of that row's video_id. Unseen-in-train videos (freq 0) always land in 'rare'."""
    f = np.array([freq.get(x[2], 0) for x in test_rows])
    q1, q2 = np.quantile(f, [1/3, 2/3])
    rare = np.where(f <= q1)[0]
    med = np.where((f > q1) & (f <= q2))[0]
    pop = np.where(f > q2)[0]
    return {'rare': rare, 'medium': med, 'popular': pop}, f


def eval_stratum(users, labels, scores, idx):
    if len(idx) == 0:
        return None
    u = [users[i] for i in idx]
    y = labels[idx]
    s = scores[idx]
    return evaluate(u, y, s)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('data_dir', nargs='?', default='./KuaiRand-Pure/data')
    ap.add_argument('--ssl_emb', default='ssl_video_emb.npy')
    ap.add_argument('--seeds', type=int, default=5, help='runs seeds 0..seeds-1')
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--epochs', type=int, default=40)
    a = ap.parse_args()

    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)
    _, field_dims, offsets = field_layout(splits['train'])
    vi = FIELDS.index('video_id')
    lo, hi = int(offsets[vi]), int(offsets[vi] + field_dims[vi])
    dim = int(sum(field_dims))

    ssl_emb = np.load(a.ssl_emb)
    expected_shape = (hi - lo, a.k)
    assert ssl_emb.shape == expected_shape, (
        f"ssl_video_emb.npy shape {ssl_emb.shape} != expected {expected_shape}")

    freq = video_train_freq(splits)
    strata, freqs = stratify_indices(splits['test'], freq)
    print(f"\ntrain video frequency terciles: q1={np.quantile(freqs,1/3):.1f}  q2={np.quantile(freqs,2/3):.1f}")
    print(f"stratum sizes: rare={len(strata['rare'])}  medium={len(strata['medium'])}  popular={len(strata['popular'])}\n")

    deltas = {'rare': [], 'medium': [], 'popular': []}
    overall_deltas = []

    for seed in range(a.seeds):
        rng = np.random.default_rng(seed)
        ssl_V = rng.normal(0, 0.01, (dim, a.k)).astype(np.float32)
        ssl_V[lo:hi] = ssl_emb

        print(f"[seed {seed}] training random-init FM ...")
        rand_out = B.run_fm(splits, k=a.k, epochs=a.epochs, seed=seed, verbose=False, return_model=True)
        print(f"[seed {seed}] training SSL warm-start FM ...")
        ssl_out = B.run_fm(splits, k=a.k, epochs=a.epochs, seed=seed, verbose=False,
                            init_V=ssl_V, return_model=True)

        row = f"  seed {seed} | "
        for name in ('rare', 'medium', 'popular'):
            idx = strata[name]
            r = eval_stratum(rand_out['test_users'], rand_out['test_labels'], rand_out['test_scores'], idx)
            s = eval_stratum(ssl_out['test_users'], ssl_out['test_labels'], ssl_out['test_scores'], idx)
            d = s['primary'] - r['primary']
            deltas[name].append(d)
            row += f"{name} {d:+.4f}  "
        od = ssl_out['test']['primary'] - rand_out['test']['primary']
        overall_deltas.append(od)
        print(row + f"overall {od:+.4f}")

    print(f"\n{'stratum':10s} | {'mean delta':>11s} | {'std':>7s} | {'min':>8s} | {'max':>8s} | consistent sign?")
    for name in ('rare', 'medium', 'popular'):
        d = np.array(deltas[name])
        consistent = "yes" if (d > 0).all() or (d < 0).all() else "no"
        print(f"{name:10s} | {d.mean():+11.4f} | {d.std():7.4f} | {d.min():+8.4f} | {d.max():+8.4f} | {consistent}")
    od = np.array(overall_deltas)
    consistent = "yes" if (od > 0).all() or (od < 0).all() else "no"
    print(f"{'overall':10s} | {od.mean():+11.4f} | {od.std():7.4f} | {od.min():+8.4f} | {od.max():+8.4f} | {consistent}")
