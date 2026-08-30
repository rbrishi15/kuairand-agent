"""Three-way comparison: random-init FM vs SSL warm-start (no freeze) vs SSL
warm-start with low-frequency video rows FROZEN (never updated after init).

Rationale: ssl_overwrite_check.py showed rare video embeddings drift MORE than
popular ones during fine-tuning (dense L2 decay + stale Adam moments hit sparse
rows hardest), yet the rare stratum still showed a small, consistent SSL benefit
despite that drift. This tests whether removing the drift entirely (by freezing
those rows at their SSL init) makes the rare-stratum benefit larger.

Prerequisite: run ssl_pretrain.py first to generate ssl_video_emb.npy.

Usage:
    python3 ssl_freeze_compare.py [data_dir] [--ssl_emb ssl_video_emb.npy] [--seeds 5]
"""
import argparse
import numpy as np
from data import load, field_layout, FIELDS
import baseline as B
from ssl_stratified import video_train_freq, stratify_indices, eval_stratum


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
    vocabs, field_dims, offsets = field_layout(splits['train'])
    vi = FIELDS.index('video_id')
    lo, hi = int(offsets[vi]), int(offsets[vi] + field_dims[vi])
    dim = int(sum(field_dims))
    video_vocab = vocabs[vi]

    ssl_emb = np.load(a.ssl_emb)
    expected_shape = (hi - lo, a.k)
    assert ssl_emb.shape == expected_shape, (
        f"ssl_video_emb.npy shape {ssl_emb.shape} != expected {expected_shape}")

    freq = video_train_freq(splits)
    strata, freqs = stratify_indices(splits['test'], freq)
    threshold = np.quantile(freqs, 1/3)  # same "rare" boundary used in ssl_stratified.py

    # absolute row indices (into the full V table) for videos at/below the rare threshold
    frozen_abs_rows = np.array([
        lo + row for vid_str, row in video_vocab.items() if freq.get(vid_str, 0) <= threshold
    ], dtype=np.int64)

    print(f"\ntrain video frequency terciles: q1={threshold:.1f}  q2={np.quantile(freqs,2/3):.1f}")
    print(f"stratum sizes: rare={len(strata['rare'])}  medium={len(strata['medium'])}  popular={len(strata['popular'])}")
    print(f"frozen video rows (freq <= {threshold:.1f}): {len(frozen_abs_rows)} / {hi-lo} videos\n")

    variants = {'random': [], 'SSL (no freeze)': [], 'SSL (freeze rare)': []}
    stratum_deltas_nofreeze = {'rare': [], 'medium': [], 'popular': []}
    stratum_deltas_freeze = {'rare': [], 'medium': [], 'popular': []}

    for seed in range(a.seeds):
        rng = np.random.default_rng(seed)
        ssl_V = rng.normal(0, 0.01, (dim, a.k)).astype(np.float32)
        ssl_V[lo:hi] = ssl_emb

        print(f"[seed {seed}] training random-init FM ...")
        r_out = B.run_fm(splits, k=a.k, epochs=a.epochs, seed=seed, verbose=False, return_model=True)
        print(f"[seed {seed}] training SSL warm-start (no freeze) FM ...")
        s_out = B.run_fm(splits, k=a.k, epochs=a.epochs, seed=seed, verbose=False,
                          init_V=ssl_V, return_model=True)
        print(f"[seed {seed}] training SSL warm-start (freeze rare) FM ...")
        f_out = B.run_fm(splits, k=a.k, epochs=a.epochs, seed=seed, verbose=False,
                          init_V=ssl_V, frozen_rows=frozen_abs_rows, return_model=True)

        variants['random'].append(r_out['test']['primary'])
        variants['SSL (no freeze)'].append(s_out['test']['primary'])
        variants['SSL (freeze rare)'].append(f_out['test']['primary'])

        row_nf = f"  seed {seed} [no freeze] | "
        row_fr = f"  seed {seed} [freeze]    | "
        for name in ('rare', 'medium', 'popular'):
            idx = strata[name]
            r = eval_stratum(r_out['test_users'], r_out['test_labels'], r_out['test_scores'], idx)
            s = eval_stratum(s_out['test_users'], s_out['test_labels'], s_out['test_scores'], idx)
            f = eval_stratum(f_out['test_users'], f_out['test_labels'], f_out['test_scores'], idx)
            d_nf = s['primary'] - r['primary']
            d_fr = f['primary'] - r['primary']
            stratum_deltas_nofreeze[name].append(d_nf)
            stratum_deltas_freeze[name].append(d_fr)
            row_nf += f"{name} {d_nf:+.4f}  "
            row_fr += f"{name} {d_fr:+.4f}  "
        print(row_nf)
        print(row_fr)

    print(f"\n{'variant':20s} | {'mean primary':>12s} | {'std':>7s}")
    for name, vals in variants.items():
        v = np.array(vals)
        print(f"{name:20s} | {v.mean():12.4f} | {v.std():7.4f}")

    print(f"\n{'stratum':10s} | {'no-freeze mean delta':>21s} | {'freeze mean delta':>18s} | freeze helped more?")
    for name in ('rare', 'medium', 'popular'):
        nf = np.array(stratum_deltas_nofreeze[name])
        fr = np.array(stratum_deltas_freeze[name])
        better = "yes" if fr.mean() > nf.mean() else "no"
        print(f"{name:10s} | {nf.mean():21.4f} | {fr.mean():18.4f} | {better}")
