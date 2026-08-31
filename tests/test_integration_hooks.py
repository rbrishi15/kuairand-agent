"""[Rishi] Regression tests for the integration wiring between Min's
DeepFMMTL hooks and Vidush's/Nandit's outputs (run_deepfm_mtl's
sample_weight/seq_arrays plumbing in src/models/deepfm_mtl.py, added on
top of Min's original model). Synthetic data, tiny and fast — these are
about the wiring, not about real KuaiRand numbers (see the ablation
configs / run_log.jsonl for those).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.models.deepfm_mtl import DeepFMMTL, run_deepfm_mtl

DIM = 100
NUM_FIELDS = 5


def _synthetic_enc(n_train=200, n_valid=50, n_test=50, seed=0):
    rng = np.random.default_rng(seed)

    def make(n):
        X = rng.integers(0, DIM, size=(n, NUM_FIELDS)).astype(np.int32)
        y = rng.integers(0, 2, size=n).astype(np.float32)
        users = [f'u{i % 10}' for i in range(n)]
        return X, y, users

    return {'train': make(n_train), 'valid': make(n_valid), 'test': make(n_test)}


def test_seq_encoder_checkpoint_does_not_leak_into_deepfm_state_dict():
    """Regression: run_deepfm_mtl used to attach the trained seq_encoder via
    plain attribute assignment (`model.seq_encoder = seq_encoder`). Because
    seq_encoder is an nn.Module, nn.Module.__setattr__ auto-registers any
    nn.Module-valued attribute as a submodule — silently folding its params
    into model.state_dict() under a "seq_encoder.*" prefix. A fresh
    DeepFMMTL() (which never had .seq_encoder set) then rejected those as
    unexpected keys on load_state_dict, breaking every seq-enabled
    checkpoint. Fixed via object.__setattr__, which this guards.
    """
    enc = _synthetic_enc()
    max_len, num_items = 5, 20
    rng = np.random.default_rng(1)
    seq_arrays = {name: rng.integers(0, num_items, size=(len(y), max_len)).astype(np.int64)
                  for name, (_, y, _) in enc.items()}
    seq_kwargs = {'num_items': num_items, 'embed_dim': 4, 'hidden_dim': 8}

    model, metrics = run_deepfm_mtl(enc, DIM, embed_dim=4, epochs=1, bs=64, seed=0,
                                     verbose=False, seq_arrays=seq_arrays, seq_kwargs=seq_kwargs)

    assert model.seq_encoder is not None
    assert 'seq_encoder' not in dict(model.named_children())
    deepfm_state = model.state_dict()
    assert not any(k.startswith('seq_encoder') for k in deepfm_state)

    # The actual regression: this must not raise "Unexpected key(s)".
    reloaded = DeepFMMTL(DIM, embed_dim=4, num_fields=NUM_FIELDS, seq_dim=model.seq_encoder.out_dim)
    reloaded.load_state_dict(deepfm_state)
    assert 'primary' in metrics['valid']


def test_run_without_seq_arrays_leaves_seq_encoder_none():
    model, metrics = run_deepfm_mtl(_synthetic_enc(), DIM, embed_dim=4, epochs=1,
                                     bs=64, seed=0, verbose=False)
    assert model.seq_encoder is None
    assert 'seq_encoder' not in dict(model.state_dict())


def test_sample_weight_is_accepted_and_training_still_runs():
    enc = _synthetic_enc()
    w = np.ones(len(enc['train'][1]), dtype=np.float32)
    model, metrics = run_deepfm_mtl(enc, DIM, embed_dim=4, epochs=1, bs=64,
                                     seed=0, verbose=False, sample_weight=w)
    assert 'primary' in metrics['valid']
    assert 'primary' in metrics['test']
