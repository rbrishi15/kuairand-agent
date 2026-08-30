"""Compare FM training objectives: pointwise logloss (the official baseline) vs
pairwise BPR vs listwise within-user softmax cross-entropy.

Motivation (SSL investigation summary, section 8): every experiment that added or
protected *information* -- static features, embedding width, SSL pretraining,
embedding freezing -- moved primary by <= noise. That points at an
objective/architecture mismatch, not a data problem: GAUC and nDCG@5 are ranking
metrics, but FM is trained pointwise. This script tests the first half of that fix
-- swap the objective, hold the FM architecture and the run_fm() training loop
(same k / lr / epochs / patience / early-stop-on-valid-primary / evaluate()) fixed
so any score gap is attributable to the loss alone.

Optionally layer SSL warm-start on top with --ssl_emb to see whether the two
directions compound.

Prerequisite for --ssl_emb: run ssl_pretrain.py first to generate ssl_video_emb.npy.

Finding (5 seeds, KuaiRand-Pure, early-stop on valid primary):
    pointwise  lr=0.001   -> test primary 0.5919 +/- 0.0013   (== official baseline path)
    bpr        lr=0.001   -> 0.5894 +/- 0.0019   (overshoots: peaks ~epoch 3, then decays)
    bpr        lr=0.0003  -> 0.5959 +/- 0.0006
    bpr        lr=0.0002  -> 0.5962 +/- 0.0005   (+0.0043 vs same-seed pointwise, ~1/2 the std)
    listwise   lr=0.001   -> 0.5922 +/- 0.0007   (ties pointwise)
BPR's per-pair gradient is larger than pointwise's (p-y)/B, so it wants ~3-5x
smaller lr; at that lr it's a real, low-variance gain over pointwise. Use --lr.

Usage:
    python3 loss_compare.py [data_dir] [--seeds 5] [--losses pointwise,bpr,listwise]
    python3 loss_compare.py [data_dir] --losses bpr --lr 0.0002      # tuned BPR
    python3 loss_compare.py [data_dir] --ssl_emb ssl_video_emb.npy   # also warm-start V
"""
import argparse, statistics
import numpy as np
from data import load, field_layout, FIELDS
import baseline as B


def summarize(name, results):
    g = statistics.mean(r['GAUC'] for r in results)
    n5 = statistics.mean(r['nDCG@5'] for r in results)
    pr = statistics.mean(r['primary'] for r in results)
    sd = statistics.pstdev([r['primary'] for r in results]) if len(results) > 1 else 0.0
    print(f"  {name:22s} | test GAUC {g:.4f} | nDCG@5 {n5:.4f} | primary {pr:.4f} +/- {sd:.4f}")
    return pr


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('data_dir', nargs='?', default='./KuaiRand-Pure/data')
    ap.add_argument('--seeds', type=int, default=3, help='runs seeds 0..seeds-1')
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--losses', default='pointwise,bpr,listwise',
                    help='comma-separated subset of pointwise,bpr,listwise')
    ap.add_argument('--ssl_emb', default=None,
                    help='optional ssl_video_emb.npy -- if given, every variant also warm-starts V')
    a = ap.parse_args()

    losses = [x.strip() for x in a.losses.split(',') if x.strip()]
    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)

    make_ssl_V = None
    if a.ssl_emb:
        _, field_dims, offsets = field_layout(splits['train'])
        vi = FIELDS.index('video_id')
        lo, hi = int(offsets[vi]), int(offsets[vi] + field_dims[vi])
        dim = int(sum(field_dims))
        ssl_emb = np.load(a.ssl_emb)
        assert ssl_emb.shape == (hi - lo, a.k), (
            f"{a.ssl_emb} shape {ssl_emb.shape} != expected {(hi - lo, a.k)} "
            f"-- regenerate with ssl_pretrain.py on this data_dir and matching --k")

        def make_ssl_V(seed):
            rng = np.random.default_rng(seed)
            V = rng.normal(0, 0.01, (dim, a.k)).astype(np.float32)
            V[lo:hi] = ssl_emb
            return V

        print(f"SSL warm-start ENABLED (video rows <- {a.ssl_emb})")

    seeds = list(range(a.seeds))
    print(f"\n{a.seeds} seed(s) x {len(losses)} loss(es), up to {a.epochs} epochs each, "
          f"early-stop patience=4\n")

    means = {}
    for ls in losses:
        print(f"loss = {ls}:")
        res = []
        for s in seeds:
            iv = make_ssl_V(s) if make_ssl_V else None
            out = B.run_fm(splits, k=a.k, lr=a.lr, epochs=a.epochs, seed=s,
                           verbose=False, loss=ls, init_V=iv)
            res.append(out['test'])
        means[ls] = summarize(ls, res)
        print()

    base = means.get('pointwise')
    if base is not None and len(means) > 1:
        print("delta primary vs pointwise:")
        for ls in losses:
            if ls != 'pointwise':
                print(f"  {ls:22s} {means[ls] - base:+.4f}")
    print("\n(official FM baseline: test primary 0.5946 +/- 0.0008 over 5 seeds -- "
          "the pointwise run above should land near it)")
