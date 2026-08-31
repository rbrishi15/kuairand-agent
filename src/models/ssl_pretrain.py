"""[Nandit] Priority 4: self-supervised pretraining of a video-embedding table.

Ported from the sequential/SSL prototype. The pretext task is **item2vec**
(skip-gram + negative sampling) over the per-user watch sequences from
``src.features.sequential.build_sequences`` — it learns ``video_id``
embeddings purely from in-sequence co-occurrence, using no ``long_view``
label. CLAUDE.md §8 frames Priority 4 as "masked-item prediction"; skip-gram
is the sibling formulation the prototype actually used (predict context items
from a target item, rather than predict a masked item from its context) and
is what the numbers on this branch were produced with.

Output: a ``(num_items + 1, k)`` float32 matrix, row-aligned to the vocab
from :func:`src.features.sequential.build_video_vocab`. Row ``num_items`` is
the PAD slot and is left at its random init (no sequence signal). The matrix
is meant to warm-start ``SeqEncoder.item_emb`` before supervised fine-tuning;
per CLAUDE.md §8 keep it only if it beats the plain multi-task model.

Pure NumPy, CPU-only (CLAUDE.md §3). Adam is used rather than plain SGD
because a batch's mean gradient here is ~1e-4; SGD at any sane lr barely
moves the embeddings, matching the reason ``src/models/fm.py`` uses Adam.
"""
import time

import numpy as np


def make_pairs(sequences, window=3):
    """Sliding-window (target, context) co-occurrence pairs.

    ``sequences``: iterable of int-id lists (values of
    :func:`src.features.sequential.build_sequences`, or any list of lists).
    Returns two aligned ``int64`` arrays ``(targets, contexts)``.
    """
    tg, ctx = [], []
    for s in sequences:
        n = len(s)
        if n < 2:
            continue
        for i, t in enumerate(s):
            lo = max(0, i - window)
            hi = min(n, i + window + 1)
            for j in range(lo, hi):
                if j == i:
                    continue
                tg.append(t)
                ctx.append(s[j])
    return np.asarray(tg, dtype=np.int64), np.asarray(ctx, dtype=np.int64)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def pretrain_item2vec(sequences, num_items, k=16, window=3, epochs=5, neg=5,
                      lr=0.01, bs=4096, seed=0, verbose=True):
    """Skip-gram + negative sampling. Returns the target-side ("In")
    embedding matrix, shape ``(num_items + 1, k)`` — the word2vec convention
    of keeping only the input embeddings. The trailing ``+1`` PAD row keeps
    its random init.

    ``sequences``: iterable of int-id lists; ids must be in ``[0, num_items)``.
    """
    if not isinstance(sequences, (list, tuple)):
        sequences = list(sequences)
    tg, ctx = make_pairs(sequences, window)
    vocab_size = num_items + 1

    rng = np.random.default_rng(seed)
    emb_in = rng.normal(0, 0.01, (vocab_size, k)).astype(np.float32)
    emb_out = rng.normal(0, 0.01, (vocab_size, k)).astype(np.float32)
    if len(tg) == 0:
        if verbose:
            print("  ssl_pretrain: no co-occurrence pairs, returning random init")
        return emb_in

    m_in = np.zeros_like(emb_in); v_in = np.zeros_like(emb_in)
    m_out = np.zeros_like(emb_out); v_out = np.zeros_like(emb_out)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t_step = 0

    # unigram^0.75 negative-sampling distribution (word2vec standard): damps
    # how often high-frequency items are drawn as negatives.
    freq = np.bincount(ctx, minlength=vocab_size).astype(np.float64)
    freq = np.maximum(freq, 1.0) ** 0.75
    neg_p = freq / freq.sum()

    n = len(tg)
    for ep in range(1, epochs + 1):
        idx = rng.permutation(n)
        t0 = time.time()
        tot_loss, nb = 0.0, 0
        for i in range(0, n, bs):
            b = idx[i:i + bs]
            tgt, c = tg[b], ctx[b]
            B = len(b)
            negs = rng.choice(vocab_size, size=(B, neg), p=neg_p)

            vt = emb_in[tgt]                                    # (B, k)
            vc = emb_out[c]                                     # (B, k)   positive
            vn = emb_out[negs]                                  # (B, neg, k) negatives

            pos_score = _sigmoid(np.sum(vt * vc, axis=1))                  # (B,)
            neg_score = _sigmoid(np.sum(vt[:, None, :] * vn, axis=2))      # (B, neg)

            loss = (-np.mean(np.log(pos_score + 1e-9))
                    - np.mean(np.log(1 - neg_score + 1e-9)))
            tot_loss += loss
            nb += 1

            g_pos = ((pos_score - 1.0) / B)[:, None]                       # (B, 1)
            g_neg = (neg_score / B)[:, :, None]                            # (B, neg, 1)

            grad_vt = g_pos * vc + np.sum(g_neg * vn, axis=1)              # (B, k)
            grad_vc = g_pos * vt                                          # (B, k)
            grad_vn = g_neg * vt[:, None, :]                              # (B, neg, k)

            g_in = np.zeros_like(emb_in)
            g_out = np.zeros_like(emb_out)
            np.add.at(g_in, tgt, grad_vt)
            np.add.at(g_out, c, grad_vc)
            np.add.at(g_out, negs.reshape(-1), grad_vn.reshape(-1, k))

            t_step += 1
            for P, G, M, V in ((emb_in, g_in, m_in, v_in),
                               (emb_out, g_out, m_out, v_out)):
                M *= b1; M += (1 - b1) * G
                V *= b2; V += (1 - b2) * (G * G)
                P -= lr * (M / (1 - b1 ** t_step)) / (np.sqrt(V / (1 - b2 ** t_step)) + eps)

        if verbose:
            print(f"  ssl epoch {ep} | loss {tot_loss / max(nb, 1):.4f} | {time.time() - t0:.1f}s")
    return emb_in
