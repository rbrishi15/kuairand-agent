"""[Rishi] Single training run: load config, train the configured model,
save a checkpoint as {state_dict, config, val_metrics} (CLAUDE.md §5).

Every model referenced in configs/*.yaml's model.name must be registered in
MODEL_REGISTRY below.

For deepfm_mtl, two optional model-config flags close the interface-contract
loop (CLAUDE.md §5, Definition of Done: hooks must actually be used, not
sit unused):
  use_ips: true  -> builds Vidush's IPS weights (src.features.propensity)
                     and threads them into DeepFMMTL's sample_weight hook.
  use_seq: true  -> builds Nandit's per-row sequence arrays
                     (src.features.sequential) and trains a SeqEncoder
                     jointly with DeepFMMTL via its seq_embedding hook.
Optional seq_max_len/seq_embed_dim/seq_hidden_dim tune the sequence side
when use_seq is set (defaults: 50/16/32). Both flags default to false, so
existing configs are unaffected.

A third flag switches the training objective:
  loss: bpr  -> pairwise BPR (README headroom idea #1) instead of pointwise
                BCE, via run_deepfm_mtl_bpr. Directly optimizes per-user
                pairwise ordering, which is what GAUC actually measures.
                use_seq composes with it; use_ips does not yet (pairwise
                IPS reweighting is its own design question) -- raises
                rather than silently ignoring one.
"""
import argparse
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.config import load_config
from src.data import load, encode, subsample_encoded
from src.models.fm import run_fm
from src.models.deepfm_mtl import run_deepfm_mtl, run_deepfm_mtl_bpr

MODEL_REGISTRY = {'fm', 'deepfm_mtl'}


def _build_ips_weights(config, splits):
    from src.features.propensity import estimate_propensity
    random_log_path = os.path.join(config['data_dir'], 'log_random_4_22_to_5_08_pure.csv')
    return estimate_propensity(splits, random_log_path)['train']


def _build_seq_arrays(splits, max_len, embed_dim, hidden_dim):
    from src.features.sequential import build_sequences, build_video_vocab, sequences_for_rows
    vocab = build_video_vocab(splits)
    user_seqs = build_sequences(splits, vocab=vocab, max_len=max_len)
    pad_id = len(vocab)
    arrays = {name: sequences_for_rows(splits[name], user_seqs, pad_id, max_len=max_len)
              for name in ('train', 'valid', 'test')}
    seq_kwargs = {'num_items': len(vocab), 'embed_dim': embed_dim, 'hidden_dim': hidden_dim}
    return arrays, seq_kwargs


def train(config, smoke_test=False, verbose=True):
    splits = load(config)
    # vocab/dim always come from the FULL train split, even in smoke-test
    # mode — see subsample_encoded()'s docstring for why order matters here.
    enc, dim = encode(splits)

    model_cfg = dict(config['model'])
    name = model_cfg.pop('name')
    if name not in MODEL_REGISTRY:
        raise ValueError(f"unknown model '{name}' — not in MODEL_REGISTRY")

    use_ips = model_cfg.pop('use_ips', False)
    use_seq = model_cfg.pop('use_seq', False)
    loss = model_cfg.pop('loss', 'pointwise')
    seq_max_len = model_cfg.pop('seq_max_len', 50)
    seq_embed_dim = model_cfg.pop('seq_embed_dim', 16)
    seq_hidden_dim = model_cfg.pop('seq_hidden_dim', 32)

    if loss == 'bpr' and use_ips:
        raise ValueError("loss='bpr' + use_ips=true not supported: pairwise IPS "
                          "reweighting needs its own design (reweight by which side "
                          "of the pair?), not attempted yet -- pick one")

    sample_weight = None
    seq_arrays = None
    seq_kwargs = None
    extra = {}

    if use_ips:
        extra.setdefault('train', {})['sample_weight'] = _build_ips_weights(config, splits)
    if use_seq:
        seq_arrays, seq_kwargs = _build_seq_arrays(splits, seq_max_len, seq_embed_dim, seq_hidden_dim)
        for split_name in ('train', 'valid', 'test'):
            extra.setdefault(split_name, {})['seq'] = seq_arrays[split_name]

    if smoke_test:
        if extra:
            enc, extra = subsample_encoded(enc, n=5000, seed=config.get('seed', 0), extra=extra)
        else:
            enc = subsample_encoded(enc, n=5000, seed=config.get('seed', 0))
        model_cfg = dict(model_cfg, epochs=1, patience=1)

    if use_ips:
        sample_weight = extra['train']['sample_weight']
    if use_seq:
        seq_arrays = {split_name: extra[split_name]['seq'] for split_name in ('train', 'valid', 'test')}

    if name == 'fm':
        model, metrics = run_fm(enc, dim, seed=config.get('seed', 0),
                                 verbose=verbose, **model_cfg)
        state_dict = model.state_dict()
    elif name == 'deepfm_mtl':
        if loss == 'bpr':
            model, metrics = run_deepfm_mtl_bpr(enc, dim, seed=config.get('seed', 0), verbose=verbose,
                                                 seq_arrays=seq_arrays, seq_kwargs=seq_kwargs, **model_cfg)
        else:
            model, metrics = run_deepfm_mtl(enc, dim, seed=config.get('seed', 0), verbose=verbose,
                                             sample_weight=sample_weight, seq_arrays=seq_arrays,
                                             seq_kwargs=seq_kwargs, **model_cfg)
        state_dict = {
            'deepfm': model.state_dict(),
            'seq_encoder': model.seq_encoder.state_dict() if model.seq_encoder is not None else None,
        }
    else:
        raise NotImplementedError(f"model '{name}' registered but no training path wired up")

    # Checkpoint's config is self-describing: includes the derived seq_kwargs
    # (num_items etc. come from the data, not the yaml) so
    # scripts/eval_checkpoint.py can rebuild the exact same architecture
    # without re-deriving anything.
    saved_config = copy.deepcopy(config)
    if use_seq:
        saved_config['model']['seq_kwargs'] = seq_kwargs

    # Variant suffix so e.g. base/+ips/+seq/+bpr runs of the same model
    # don't overwrite each other's checkpoint (they share name + seed).
    variant = name + ('-bpr' if loss == 'bpr' else '') + ('-ips' if use_ips else '') + ('-seq' if use_seq else '')
    checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_name = 'smoke_test.pt' if smoke_test else f"{variant}_seed{config.get('seed', 0)}.pt"
    ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
    torch.save({'state_dict': state_dict, 'config': saved_config, 'val_metrics': metrics['valid']},
               ckpt_path)

    return {'checkpoint_path': ckpt_path, 'val_metrics': metrics['valid'],
            'test_metrics': metrics['test']}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--smoke-test', action='store_true')
    ap.add_argument('--seed', type=int, default=None,
                     help='override config["seed"], e.g. for a multi-seed sweep')
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.seed is not None:
        cfg['seed'] = a.seed
    result = train(cfg, smoke_test=a.smoke_test)
    print(f"checkpoint: {result['checkpoint_path']}")
    v = result['val_metrics']
    print(f"valid  GAUC {v['GAUC']:.4f} | nDCG@5 {v['nDCG@5']:.4f} | primary {v['primary']:.4f}")
