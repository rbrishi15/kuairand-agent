"""[Nandit] Scratch runner for Priority 3/4 local iterations, pre-merge.

CLAUDE.md §6 allows any role to run their own logged iterations before
merging. This is that: it trains Min's ``DeepFMMTL`` with a ``SeqEncoder``
(this branch's Priority 3) attached through the ``seq_embedding`` hook, on
the frozen data/eval path (``src.data`` / ``src.evaluate``), and appends one
``logs/run_log.jsonl`` entry per run in the §7 schema with ``role: "Nandit"``.

It deliberately does **not** touch ``src/train.py`` (Rishi's file) or
``src/models/deepfm_mtl.py`` (Min's). Rishi wires a ``deepfm_mtl_seq`` path
into ``src/train.py``'s ``MODEL_REGISTRY`` at integration time; until then
this stands in.

Examples
--------
    # Priority 3: DeepFMMTL + GRU sequence encoder
    python experiments/run_seq.py --config configs/kuairand_pure_deepfm_mtl.yaml

    # ablation baseline: same loop, seq encoder disabled (seq_dim=0)
    python experiments/run_seq.py --no-seq

    # Priority 4: warm-start the seq encoder's item table with item2vec
    python experiments/run_seq.py --ssl-warmstart --ssl-epochs 5
"""
import argparse
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from agent.orchestrator import append_log, read_log
from src.config import load_config
from src.data import encode, load
from src.evaluate import evaluate
from src.features.sequential import (build_sequences, build_video_vocab,
                                     sequences_for_rows)
from src.models.deepfm_mtl import DeepFMMTL
from src.models.seq_encoder import SeqEncoder
from src.models.ssl_pretrain import pretrain_item2vec

DEFAULT_CONFIG = 'configs/kuairand_pure_deepfm_mtl.yaml'


def _seq_arrays(splits, enc, seq_len):
    """Per-row left-padded id sequences for train/valid/test, plus the vocab
    size (== PAD id)."""
    vocab = build_video_vocab(splits)
    user_seqs = build_sequences(splits, vocab=vocab, max_len=seq_len)
    pad_id = len(vocab)
    arrays = {name: sequences_for_rows(splits[name], user_seqs, pad_id, max_len=seq_len)
              for name in ('train', 'valid', 'test')}
    return arrays, vocab, user_seqs, pad_id


def train_seq(config, seq_len=50, seq_hidden=32, pooling='gru', use_seq=True,
              ssl_warmstart=False, ssl_epochs=5, verbose=True):
    mcfg = dict(config['model'])
    mcfg.pop('name', None)
    embed_dim = int(mcfg.get('embed_dim', 16))
    lr = float(mcfg.get('lr', 1e-3))
    bs = int(mcfg.get('bs', 8192))
    epochs = int(mcfg.get('epochs', 40))
    patience = int(mcfg.get('patience', 4))
    seed = int(config.get('seed', 0))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed)

    splits = load(config)
    enc, dim = encode(splits)
    Xtr, ytr, _ = enc['train']
    Xva, yva, uva = enc['valid']
    Xte, yte, ute = enc['test']

    seq_encoder = None
    seq_t = {'train': None, 'valid': None, 'test': None}
    if use_seq:
        arrays, vocab, user_seqs, pad_id = _seq_arrays(splits, enc, seq_len)
        seq_t = {k: torch.from_numpy(v).long().to(device) for k, v in arrays.items()}

        pretrained = None
        if ssl_warmstart:
            if verbose:
                print(f"item2vec pretrain: {len(user_seqs)} user sequences, "
                      f"vocab {len(vocab)}, {ssl_epochs} epochs")
            pretrained = pretrain_item2vec(
                list(user_seqs.values()), num_items=len(vocab), k=embed_dim,
                epochs=ssl_epochs, seed=seed, verbose=verbose)

        seq_encoder = SeqEncoder(num_items=len(vocab), embed_dim=embed_dim,
                                 hidden_dim=seq_hidden, pooling=pooling,
                                 pretrained_emb=pretrained).to(device)

    seq_dim = seq_encoder.out_dim if seq_encoder is not None else 0
    model = DeepFMMTL(dim, embed_dim=embed_dim, num_fields=Xtr.shape[1],
                      seq_dim=seq_dim).to(device)

    params = list(model.parameters())
    if seq_encoder is not None:
        params += list(seq_encoder.parameters())
    opt = torch.optim.Adam(params, lr=lr)

    Xtr_t = torch.from_numpy(Xtr).long().to(device)
    ytr_t = torch.from_numpy(ytr).float().to(device)
    Xva_t = torch.from_numpy(Xva).long().to(device)
    Xte_t = torch.from_numpy(Xte).long().to(device)

    def predict(X_t, seq_split, chunk=200_000):
        model.eval()
        if seq_encoder is not None:
            seq_encoder.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X_t), chunk):
                se = None
                if seq_encoder is not None:
                    se = seq_encoder(seq_split[i:i + chunk])
                out.append(model(X_t[i:i + chunk], seq_embedding=se)['primary'].cpu().numpy())
        return np.concatenate(out)

    best, best_state, bad = -1.0, None, 0
    n = len(ytr)
    for ep in range(1, epochs + 1):
        model.train()
        if seq_encoder is not None:
            seq_encoder.train()
        t0 = time.time()
        idx = torch.randperm(n, device=device)
        losses = []
        for i in range(0, n, bs):
            b = idx[i:i + bs]
            se = seq_encoder(seq_t['train'][b]) if seq_encoder is not None else None
            out = model(Xtr_t[b], seq_embedding=se)
            loss = model.compute_loss(out, ytr_t[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        va = evaluate(uva, yva, predict(Xva_t, seq_t['valid']))
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid "
                  f"GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} "
                  f"primary {va['primary']:.4f} | {time.time() - t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = ({k: v.detach().clone() for k, v in model.state_dict().items()},
                          None if seq_encoder is None
                          else {k: v.detach().clone() for k, v in seq_encoder.state_dict().items()})
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"  early stop at epoch {ep}")
                break

    model.load_state_dict(best_state[0])
    if seq_encoder is not None:
        seq_encoder.load_state_dict(best_state[1])

    return {'valid': evaluate(uva, yva, predict(Xva_t, seq_t['valid'])),
            'test': evaluate(ute, yte, predict(Xte_t, seq_t['test']))}


def _model_tag(use_seq, pooling, ssl_warmstart):
    if not use_seq:
        return 'deepfm_mtl'
    tag = f'deepfm_mtl+seq_{pooling}'
    if ssl_warmstart:
        tag += '+ssl'
    return tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=DEFAULT_CONFIG)
    ap.add_argument('--seq-len', type=int, default=50)
    ap.add_argument('--seq-hidden', type=int, default=32)
    ap.add_argument('--pooling', choices=['gru', 'mean'], default='gru')
    ap.add_argument('--no-seq', action='store_true',
                    help='ablation: run the same loop with the seq encoder disabled')
    ap.add_argument('--ssl-warmstart', action='store_true',
                    help='pretrain the seq encoder item table with item2vec first')
    ap.add_argument('--ssl-epochs', type=int, default=5)
    ap.add_argument('--role', default='Nandit')
    ap.add_argument('--hypothesis', default=None,
                    help='one-sentence §7 hypothesis; a default is filled in per mode')
    ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args()

    config = load_config(a.config)
    use_seq = not a.no_seq
    log_path = config.get('log_path', 'logs/run_log.jsonl')
    dataset = config.get('dataset')
    model_tag = _model_tag(use_seq, a.pooling, a.ssl_warmstart)

    hypothesis = a.hypothesis or (
        "Baseline for the sequence experiments: DeepFMMTL with seq_embedding "
        "disabled, same runner/loop, to measure the seq encoder's delta against."
        if not use_seq else
        f"Attach a {a.pooling} SeqEncoder over each user's train-window video "
        f"history (len {a.seq_len}, hidden {a.seq_hidden}) to DeepFMMTL via the "
        f"seq_embedding hook"
        + ("; warm-start its item table with item2vec (Priority 4)" if a.ssl_warmstart else "")
        + "; expect a primary lift from behavioural history the static fields don't carry.")

    history = read_log(log_path)
    iteration = (history[-1]['iteration'] if history else 0) + 1
    prior_same_ds = [h['metrics']['primary'] for h in history
                     if h.get('dataset') == dataset and h.get('metrics')]
    prev_best = max(prior_same_ds) if prior_same_ds else None

    t0 = time.time()
    error, recovery, metrics = None, None, None
    try:
        res = train_seq(config, seq_len=a.seq_len, seq_hidden=a.seq_hidden,
                        pooling=a.pooling, use_seq=use_seq,
                        ssl_warmstart=a.ssl_warmstart, ssl_epochs=a.ssl_epochs,
                        verbose=not a.quiet)
        metrics = res['valid']
        test_metrics = res['test']
    except Exception as e:  # CLAUDE.md §6: log the failure + recovery, don't halt
        error = f"{type(e).__name__}: {e}"
        recovery = 'logged failure; no checkpoint written; config left unchanged'
        test_metrics = None

    entry = {
        'iteration': iteration,
        'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'role': a.role,
        'dataset': dataset,
        'hypothesis': hypothesis,
        'code_diff_ref': None,
        'model': model_tag,
        'metrics': metrics,
        'delta_vs_prev_best': (metrics['primary'] - prev_best)
                              if metrics and prev_best is not None else None,
        'error': error,
        'recovery_action': recovery,
        'manual_intervention': False,
        'wall_clock_sec': round(time.time() - t0, 1),
        'tokens_used': {'input': 0, 'output': 0},
    }
    append_log(log_path, entry)

    if error:
        print(f"iter {iteration} [{model_tag}] FAILED: {error}")
        return
    d = entry['delta_vs_prev_best']
    print(f"iter {iteration} [{model_tag}] valid primary {metrics['primary']:.4f} "
          f"(GAUC {metrics['GAUC']:.4f} / nDCG@5 {metrics['nDCG@5']:.4f})"
          + (f" | delta vs prev best {d:+.4f}" if d is not None else ""))
    if test_metrics:
        print(f"           test primary {test_metrics['primary']:.4f} "
              f"(GAUC {test_metrics['GAUC']:.4f} / nDCG@5 {test_metrics['nDCG@5']:.4f})")


if __name__ == '__main__':
    main()
