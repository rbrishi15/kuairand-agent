"""[Rishi] Load a checkpoint and independently re-score it on a split.

scripts/check.sh needs a CLI that evaluates a saved checkpoint, but
src/evaluate.py is starter-kit provided and explicitly DO NOT MODIFY
(CLAUDE.md §4) — it has no CLI, only the pure evaluate() function. This
script lives outside src/evaluate.py for exactly that reason: it calls the
untouched evaluate() rather than adding a CLI on top of the frozen file.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.data import load, load_random_log, encode
from src.evaluate import evaluate
from src.scoring import score


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--split', default='valid', choices=['valid', 'test', 'offpolicy'],
                     help="'offpolicy' scores against the randomized-exposure log restricted "
                          "to the valid window (src.data.load_random_log) -- a genuine check "
                          "of whether IPS helps, since 'valid' can't reward it (see README).")
    a = ap.parse_args()

    ckpt = torch.load(a.checkpoint, weights_only=False)
    config = ckpt['config']
    splits = load(config)
    if a.split == 'offpolicy':
        splits['offpolicy'] = load_random_log(config)
    enc, dim = encode(splits)
    X, y, u = enc[a.split]

    scores = score(config, ckpt['state_dict'], X, dim, splits=splits, split_name=a.split)
    metrics = evaluate(u, y, scores)
    print(f"{a.split}  GAUC {metrics['GAUC']:.4f} | nDCG@5 {metrics['nDCG@5']:.4f} "
          f"| primary {metrics['primary']:.4f}")
