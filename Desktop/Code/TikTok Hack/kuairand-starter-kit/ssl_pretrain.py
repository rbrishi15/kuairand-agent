"""SSL 预训练：item2vec（skip-gram + negative sampling），只用观看序列的共现关系
学出 video_id 的 embedding —— 不用 long_view 标签。

只用 train 时间窗口内的数据（避免向 valid/test 泄漏），按 (user_id, date, hourmin)
排出每个用户的观看序列，用序列内的共现关系训练。

用法：
    python3 ssl_pretrain.py [data_dir] [--k 16] [--window 3] [--epochs 5] [--neg 5]

输出：
    ssl_video_emb.npy —— shape (video_vocab_size + 1, k)。
    行号与 data.py::build_vocabs() 给出的 video_id 词表 id 完全对齐（含 UNK 槽），
    可以直接切片赋给 baseline.py 里 FM.V 对应 video_id 的那一段。
"""
import argparse, csv, os, time
import numpy as np
from data import load, build_vocabs, FIELDS, SPLITS


def build_sequences(data_dir, video_vocab):
    """只读 train 窗口内的日志，按用户分组、按时间排序，返回 dict: user_id -> [video_vocab_id, ...]。"""
    lo, hi = SPLITS['train']
    seqs = {}
    for fname in ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv'):
        path = os.path.join(data_dir, fname)
        with open(path) as fh:
            reader = csv.DictReader(fh)
            has_hourmin = reader.fieldnames is not None and 'hourmin' in reader.fieldnames
            for r in reader:
                d = int(r['date'])
                if not (lo <= d <= hi):
                    continue
                vid = r['video_id']
                vidx = video_vocab.get(vid)
                if vidx is None:
                    continue  # 理论上不会发生：词表本身就是从 train 建的
                hm = int(r['hourmin']) if has_hourmin else 0
                seqs.setdefault(r['user_id'], []).append((d, hm, vidx))
    for u in seqs:
        seqs[u].sort(key=lambda t: (t[0], t[1]))
        seqs[u] = [v for _, _, v in seqs[u]]
    return seqs


def make_pairs(seqs, window):
    """滑窗生成 (target, context) 共现对。"""
    tg, ctx = [], []
    for s in seqs.values():
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


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def train_skipgram(tg, ctx, vocab_size, k=16, epochs=5, neg=5, lr=0.01, bs=4096, seed=0, verbose=True):
    """skip-gram + negative sampling，Adam 优化（和 baseline.py 里 FM.step() 用的是同一套
    更新规则）。两套 embedding：In（target 用）/ Out（context 用）；最终只取 In 作为要
    注入 FM 的 video embedding（word2vec 的常规约定）。

    用 Adam 而不是普通 SGD 是因为：单个 batch 的平均梯度量级很小（~1e-4 级），普通 SGD
    在合理的 lr 下几乎不会移动 embedding；Adam 按每个参数自己的梯度历史自适应步长，
    对这种量级不敏感，训练更稳定 —— 和 FM 用 Adam 的原因完全一样。
    """
    rng = np.random.default_rng(seed)
    In = rng.normal(0, 0.01, (vocab_size, k)).astype(np.float32)
    Out = rng.normal(0, 0.01, (vocab_size, k)).astype(np.float32)
    mIn = np.zeros_like(In); vIn = np.zeros_like(In)
    mOut = np.zeros_like(Out); vOut = np.zeros_like(Out)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t_step = 0

    # unigram^0.75 负采样分布：word2vec 标准做法，压低高频 item 被过度采样的概率
    freq = np.bincount(ctx, minlength=vocab_size).astype(np.float64)
    freq = np.maximum(freq, 1.0) ** 0.75
    neg_p = freq / freq.sum()

    n = len(tg)
    for ep in range(1, epochs + 1):
        idx = rng.permutation(n)
        t0 = time.time()
        tot_loss = 0.0
        nb = 0
        for i in range(0, n, bs):
            b = idx[i:i + bs]
            tgt, c = tg[b], ctx[b]
            B = len(b)
            negs = rng.choice(vocab_size, size=(B, neg), p=neg_p)

            vt = In[tgt]                                  # (B,k)
            vc = Out[c]                                   # (B,k)   正样本
            vn = Out[negs]                                # (B,neg,k) 负样本

            pos_score = sigmoid(np.sum(vt * vc, axis=1))                 # (B,)
            neg_score = sigmoid(np.sum(vt[:, None, :] * vn, axis=2))     # (B,neg)

            loss = -np.mean(np.log(pos_score + 1e-9)) - np.mean(np.log(1 - neg_score + 1e-9))
            tot_loss += loss
            nb += 1

            g_pos = ((pos_score - 1.0) / B)[:, None]                     # (B,1)
            g_neg = (neg_score / B)[:, :, None]                          # (B,neg,1)

            grad_vt = g_pos * vc + np.sum(g_neg * vn, axis=1)            # (B,k)
            grad_vc = g_pos * vt                                         # (B,k)
            grad_vn = g_neg * vt[:, None, :]                             # (B,neg,k)

            gIn = np.zeros_like(In)
            gOut = np.zeros_like(Out)
            np.add.at(gIn, tgt, grad_vt)
            np.add.at(gOut, c, grad_vc)
            np.add.at(gOut, negs.reshape(-1), grad_vn.reshape(-1, k))

            t_step += 1
            for P, G, M, Vv in ((In, gIn, mIn, vIn), (Out, gOut, mOut, vOut)):
                M *= b1; M += (1 - b1) * G
                Vv *= b2; Vv += (1 - b2) * (G * G)
                P -= lr * (M / (1 - b1 ** t_step)) / (np.sqrt(Vv / (1 - b2 ** t_step)) + eps)
        if verbose:
            print(f"  epoch {ep} | loss {tot_loss / max(nb,1):.4f} | {time.time()-t0:.1f}s")
    return In


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('data_dir', nargs='?', default='./KuaiRand-Pure/data')
    ap.add_argument('--k', type=int, default=16, help='要和 FM 的 k 保持一致')
    ap.add_argument('--window', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--neg', type=int, default=5)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--bs', type=int, default=4096)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='ssl_video_emb.npy')
    a = ap.parse_args()

    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)
    vocabs, edges, _ = build_vocabs(splits['train'])
    video_vocab = vocabs[FIELDS.index('video_id')]
    vocab_size = len(video_vocab) + 1   # +1 UNK 槽，和 encode() 的 field_dims 对齐

    print(f"video vocab (train-only): {len(video_vocab)} + 1 UNK = {vocab_size}")
    seqs = build_sequences(a.data_dir, video_vocab)
    lens = [len(s) for s in seqs.values()]
    print(f"users with sequences: {len(seqs)} | avg len {np.mean(lens):.1f} | median {int(np.median(lens))}")

    tg, ctx = make_pairs(seqs, a.window)
    print(f"skip-gram pairs: {len(tg):,}")

    emb = train_skipgram(tg, ctx, vocab_size, k=a.k, epochs=a.epochs, neg=a.neg,
                          lr=a.lr, bs=a.bs, seed=a.seed)
    # UNK 槽（最后一行）没有序列信号，保持随机初始化
    np.save(a.out, emb)
    print(f"saved {a.out}  shape={emb.shape}")
