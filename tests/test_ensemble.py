"""[Rishi] Tests for src/models/ensemble.py's rank-average blending."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from src.models.ensemble import _percentile_rank_within_user, ensemble_predict


def test_percentile_rank_within_a_single_user_group():
    scores = np.array([3.0, 1.0, 2.0])
    users = np.array(['u0', 'u0', 'u0'])
    ranks = _percentile_rank_within_user(scores, users)
    # 1.0 -> rank 0 -> 0/2=0.0; 2.0 -> rank 1 -> 1/2=0.5; 3.0 -> rank 2 -> 2/2=1.0
    assert np.allclose(sorted(ranks), [0.0, 0.5, 1.0])
    assert ranks[0] == 1.0 and ranks[1] == 0.0 and ranks[2] == 0.5


def test_percentile_rank_is_computed_independently_per_user():
    scores = np.array([10.0, 20.0, 1.0, 2.0])
    users = np.array(['u0', 'u0', 'u1', 'u1'])
    ranks = _percentile_rank_within_user(scores, users)
    # within u0: 10<20 -> [0,1]; within u1: 1<2 -> [0,1] -- absolute scale (10s vs 1s) must not matter
    assert ranks[0] == 0.0 and ranks[1] == 1.0
    assert ranks[2] == 0.0 and ranks[3] == 1.0


def test_lone_row_user_gets_half():
    scores = np.array([5.0, 1.0, 2.0])
    users = np.array(['solo', 'u1', 'u1'])
    ranks = _percentile_rank_within_user(scores, users)
    assert ranks[0] == 0.5


def test_ensemble_predict_rejects_mismatched_weights():
    with pytest.raises(ValueError, match='must match'):
        ensemble_predict(['a.pt', 'b.pt'], np.zeros((2, 5)), 10, ['u0', 'u1'], weights=[1.0])


def test_ensemble_predict_rejects_empty_checkpoint_list():
    with pytest.raises(ValueError, match='not be empty'):
        ensemble_predict([], np.zeros((0, 5)), 10, [])
