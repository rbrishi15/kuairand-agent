"""[Rishi, on Sarthak's behalf] Evaluate a rank-average ensemble of
checkpoints on a split. Complements scripts/eval_checkpoint.py (single
checkpoint) — this is the multi-checkpoint counterpart src/models/ensemble.py
needs, since eval_checkpoint.py only ever scored one at a time.

Usage:
    python scripts/eval_ensemble.py --split valid \
        --checkpoint checkpoints/deepfm_mtl_seed0.pt --weight 0.1 \
        --checkpoint checkpoints/deepfm_mtl_seed1.pt --weight 0.1 \
        ...

All checkpoints must share the same dim (i.e. come from the same
dataset/config) — asserted, not assumed.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.data import load, load_random_log, encode
from src.evaluate import evaluate
from src.models.ensemble import ensemble_predict


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', action='append', required=True, dest='checkpoints')
    ap.add_argument('--weight', action='append', type=float, dest='weights',
                     help='one per --checkpoint, same order; omit for uniform weights')
    ap.add_argument('--split', default='valid', choices=['valid', 'test', 'offpolicy'])
    a = ap.parse_args()

    if a.weights and len(a.weights) != len(a.checkpoints):
        raise SystemExit(f'{len(a.weights)} --weight flags but {len(a.checkpoints)} --checkpoint flags')

    # Use the first checkpoint's saved config as the shared data/config
    # source -- asserted below that every checkpoint agrees on dim.
    first_config = torch.load(a.checkpoints[0], weights_only=False)['config']
    splits = load(first_config)
    if a.split == 'offpolicy':
        splits['offpolicy'] = load_random_log(first_config)
    enc, dim = encode(splits)
    X, y, u = enc[a.split]

    for path in a.checkpoints:
        cfg = torch.load(path, weights_only=False)['config']
        if (cfg['data_dir'], cfg['splits']) != (first_config['data_dir'], first_config['splits']):
            raise SystemExit(f"{path}: data_dir/splits differ from {a.checkpoints[0]} -- "
                              f"checkpoints must share the same dataset to ensemble "
                              f"(dim is derived from the same train split for all of them)")

    scores = ensemble_predict(a.checkpoints, X, dim, u, splits=splits, split_name=a.split,
                               weights=a.weights)
    metrics = evaluate(u, y, scores)
    print(f"ensemble ({len(a.checkpoints)} checkpoints) {a.split}  "
          f"GAUC {metrics['GAUC']:.4f} | nDCG@5 {metrics['nDCG@5']:.4f} | primary {metrics['primary']:.4f}")
