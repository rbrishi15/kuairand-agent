"""[Rishi] Single training run: load config, train the configured model,
save a checkpoint as {state_dict, config, val_metrics} (CLAUDE.md §5).

Every model referenced in configs/*.yaml's model.name must be registered in
MODEL_REGISTRY below. Only 'fm' exists on Day 1 — Min/Nandit register their
own here once ready.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.config import load_config
from src.data import load, encode, subsample_encoded
from src.models.fm import run_fm
from src.models.deepfm_mtl import run_deepfm_mtl

MODEL_REGISTRY = {'fm', 'deepfm_mtl'}


def train(config, smoke_test=False, verbose=True):
    splits = load(config)
    # vocab/dim always come from the FULL train split, even in smoke-test
    # mode — see subsample_encoded()'s docstring for why order matters here.
    enc, dim = encode(splits)
    if smoke_test:
        enc = subsample_encoded(enc, n=5000, seed=config.get('seed', 0))

    model_cfg = dict(config['model'])
    name = model_cfg.pop('name')
    if name not in MODEL_REGISTRY:
        raise ValueError(f"unknown model '{name}' — not in MODEL_REGISTRY")

    if smoke_test:
        model_cfg = dict(model_cfg, epochs=1, patience=1)

    if name == 'fm':
        model, metrics = run_fm(enc, dim, seed=config.get('seed', 0),
                                 verbose=verbose, **model_cfg)
        state_dict = model.state_dict()
    elif name == 'deepfm_mtl':
        model, metrics = run_deepfm_mtl(enc, dim, seed=config.get('seed', 0),
                                         verbose=verbose, **model_cfg)
        state_dict = model.state_dict()
    else:
        raise NotImplementedError(f"model '{name}' registered but no training path wired up")

    checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_name = 'smoke_test.pt' if smoke_test else f"{name}_seed{config.get('seed', 0)}.pt"
    ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
    torch.save({'state_dict': state_dict, 'config': config, 'val_metrics': metrics['valid']},
               ckpt_path)

    return {'checkpoint_path': ckpt_path, 'val_metrics': metrics['valid'],
            'test_metrics': metrics['test']}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--smoke-test', action='store_true')
    a = ap.parse_args()
    cfg = load_config(a.config)
    result = train(cfg, smoke_test=a.smoke_test)
    print(f"checkpoint: {result['checkpoint_path']}")
    v = result['val_metrics']
    print(f"valid  GAUC {v['GAUC']:.4f} | nDCG@5 {v['nDCG@5']:.4f} | primary {v['primary']:.4f}")
