"""[Rishi] Tests for BPR pairwise training (src/models/deepfm_mtl.py's
run_deepfm_mtl_bpr and its pair-sampling helpers). Synthetic data, tiny and
fast — about correctness of the sampling/training wiring, not real numbers.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.models.deepfm_mtl import (
    DeepFMMTL,
    _build_bpr_index,
    _sample_bpr_pairs,
    run_deepfm_mtl_bpr,
)

DIM = 100
NUM_FIELDS = 5


def test_build_bpr_index_excludes_single_class_users():
    # user 0: both classes (eligible). user 1: positive-only. user 2: negative-only.
    y = np.array([1, 0, 1, 1, 0], dtype=np.float32)
    users = ['u0', 'u0', 'u1', 'u1', 'u2']
    all_pos_idx, all_pos_code, neg_by_code = _build_bpr_index(y, users)
    assert set(all_pos_idx.tolist()) == {0}  # only u0's positive row (index 0)
    assert len(neg_by_code) == 1
    assert neg_by_code[all_pos_code[0]].tolist() == [1]  # u0's negative row


def test_build_bpr_index_raises_with_no_eligible_users():
    y = np.array([1, 1, 0, 0], dtype=np.float32)
    users = ['u0', 'u0', 'u1', 'u1']  # nobody has both classes
    try:
        _build_bpr_index(y, users)
        assert False, 'expected ValueError'
    except ValueError:
        pass


def test_sample_bpr_pairs_only_pairs_within_the_same_user():
    y = np.array([1, 0, 1, 0, 1, 0], dtype=np.float32)
    users = ['u0', 'u0', 'u1', 'u1', 'u1', 'u1']
    all_pos_idx, all_pos_code, neg_by_code = _build_bpr_index(y, users)
    rng = np.random.default_rng(0)
    pos_idx, neg_idx = _sample_bpr_pairs(all_pos_idx, all_pos_code, neg_by_code, 200, rng)
    users_arr = np.array(users)
    assert (users_arr[pos_idx] == users_arr[neg_idx]).all()
    assert (y[pos_idx] == 1).all()
    assert (y[neg_idx] == 0).all()


def _synthetic_enc(n_train=300, n_valid=50, n_test=50, seed=0):
    rng = np.random.default_rng(seed)

    def make(n, n_users=10):
        X = rng.integers(0, DIM, size=(n, NUM_FIELDS)).astype(np.int32)
        y = rng.integers(0, 2, size=n).astype(np.float32)
        users = [f'u{i % n_users}' for i in range(n)]
        return X, y, users

    return {'train': make(n_train), 'valid': make(n_valid), 'test': make(n_test)}


def test_run_deepfm_mtl_bpr_trains_and_saves_a_loadable_checkpoint():
    model, metrics = run_deepfm_mtl_bpr(_synthetic_enc(), DIM, embed_dim=4, epochs=2,
                                         bs=64, seed=0, verbose=False)
    assert model.seq_encoder is None
    assert 'primary' in metrics['valid']
    reloaded = DeepFMMTL(DIM, embed_dim=4, num_fields=NUM_FIELDS)
    reloaded.load_state_dict(model.state_dict())  # must not raise


def test_run_deepfm_mtl_bpr_with_seq_arrays_does_not_leak_into_state_dict():
    enc = _synthetic_enc()
    max_len, num_items = 5, 20
    rng = np.random.default_rng(1)
    seq_arrays = {name: rng.integers(0, num_items, size=(len(y), max_len)).astype(np.int64)
                  for name, (_, y, _) in enc.items()}
    seq_kwargs = {'num_items': num_items, 'embed_dim': 4, 'hidden_dim': 8}

    model, metrics = run_deepfm_mtl_bpr(enc, DIM, embed_dim=4, epochs=2, bs=64, seed=0,
                                         verbose=False, seq_arrays=seq_arrays, seq_kwargs=seq_kwargs)
    assert model.seq_encoder is not None
    deepfm_state = model.state_dict()
    assert not any(k.startswith('seq_encoder') for k in deepfm_state)
    reloaded = DeepFMMTL(DIM, embed_dim=4, num_fields=NUM_FIELDS, seq_dim=model.seq_encoder.out_dim)
    reloaded.load_state_dict(deepfm_state)  # must not raise
