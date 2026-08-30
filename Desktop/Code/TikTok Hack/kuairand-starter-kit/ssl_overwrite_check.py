"""Check whether supervised fine-tuning overwrites the SSL-pretrained video
embeddings, and whether that overwriting correlates with train frequency.

For the SSL warm-start model: compare each video's FINAL embedding (after FM
training) to its INITIAL SSL embedding. If rare videos keep high cosine
similarity to their SSL starting point (few gradient touches -> little movement)
while popular videos drift toward low similarity (many gradient touches -> mostly
overwritten), that's a direct, mechanistic explanation for why SSL's benefit
showed up only in the rare stratum.

Prerequisite: run ssl_pretrain.py first to generate ssl_video_emb.npy.

Usage:
    python3 ssl_overwrite_check.py [data_dir] [--ssl_emb ssl_video_emb.npy] [--seed 0]
"""
import argparse, collections
import numpy as np
from data import load, field_layout, FIELDS
import baseline as B


def video_train_freq(splits):
    return collections.Counter(x[2] for x in splits['train'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('data_dir', nargs='?', default='./KuaiRand-Pure/data')
    ap.add_argument('--ssl_emb', default='ssl_video_emb.npy')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--epochs', type=int, default=40)
    a = ap.parse_args()

    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)
    vocabs, field_dims, offsets = field_layout(splits['train'])
    vi = FIELDS.index('video_id')
    lo, hi = int(offsets[vi]), int(offsets[vi] + field_dims[vi])
    dim = int(sum(field_dims))
    video_vocab = vocabs[vi]                          # video_id str -> row index (0-based within field)
    idx_to_vid = {v: k for k, v in video_vocab.items()}

    ssl_emb = np.load(a.ssl_emb)
    expected_shape = (hi - lo, a.k)
    assert ssl_emb.shape == expected_shape, (
        f"ssl_video_emb.npy shape {ssl_emb.shape} != expected {expected_shape}")

    rng = np.random.default_rng(a.seed)
    ssl_V = rng.normal(0, 0.01, (dim, a.k)).astype(np.float32)
    ssl_V[lo:hi] = ssl_emb

    print(f"training SSL warm-start FM (seed={a.seed}) ...")
    out = B.run_fm(splits, k=a.k, epochs=a.epochs, seed=a.seed, verbose=False,
                    init_V=ssl_V, return_model=True)
    final_V = out['model'].V[lo:hi]                    # trained video embeddings, same row order as ssl_emb

    freq = video_train_freq(splits)
    row_freq = np.array([freq.get(idx_to_vid.get(r, None), 0) for r in range(hi - lo)])

    # cosine similarity between each video's initial SSL embedding and its final (trained) embedding
    def row_norm(x):
        n = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.maximum(n, 1e-9)

    init_n = row_norm(ssl_emb)
    final_n = row_norm(final_V)
    cos_sim = np.sum(init_n * final_n, axis=1)          # (n_videos,)
    l2_move = np.linalg.norm(final_V - ssl_emb, axis=1)  # (n_videos,)

    q1, q2 = np.quantile(row_freq, [1/3, 2/3])
    buckets = {
        'rare':    row_freq <= q1,
        'medium':  (row_freq > q1) & (row_freq <= q2),
        'popular': row_freq > q2,
    }

    print(f"\nvideo train-frequency terciles: q1={q1:.1f}  q2={q2:.1f}\n")
    print(f"{'stratum':10s} | {'n videos':>9s} | {'avg freq':>9s} | {'cos(init,final)':>16s} | {'L2 movement':>12s}")
    for name, mask in buckets.items():
        n = mask.sum()
        print(f"{name:10s} | {n:9d} | {row_freq[mask].mean():9.1f} | "
              f"{cos_sim[mask].mean():16.4f} | {l2_move[mask].mean():12.4f}")

    corr = np.corrcoef(row_freq, l2_move)[0, 1]
    print(f"\ncorrelation(train frequency, L2 movement from SSL init) = {corr:+.3f}")
    print("(positive correlation = more-frequently-seen videos move further from their SSL")
    print(" starting point during training -- i.e. gradient descent overwrites SSL structure")
    print(" roughly in proportion to how many supervised updates that video's row received)")
    