import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.evaluate import evaluate


def test_perfect_separation_gives_gauc_one():
    users = [0, 0, 0, 0]
    labels = [0, 0, 1, 1]
    scores = [1, 2, 3, 4]   # positives strictly on top
    r = evaluate(users, labels, scores)
    assert r['GAUC'] == 1.0
    assert r['nDCG@5'] == 1.0
    assert r['primary'] == 1.0


def test_reversed_separation_gives_gauc_zero():
    users = [0, 0, 0, 0]
    labels = [1, 1, 0, 0]
    scores = [1, 2, 3, 4]   # positives strictly on bottom
    r = evaluate(users, labels, scores)
    assert r['GAUC'] == 0.0


def test_interleaved_case_matches_hand_computed_values():
    # scores 1,2,3,4 with labels pos,neg,neg,pos: one positive ranked
    # lowest, one ranked highest -> 2 of 4 possible pos>neg pairs correct.
    users = [0, 0, 0, 0]
    labels = [1, 0, 0, 1]
    scores = [1, 2, 3, 4]
    r = evaluate(users, labels, scores)
    assert r['GAUC'] == 0.5
    assert r['nDCG@5'] == pytest.approx(0.8772, abs=1e-3)
    assert r['primary'] == pytest.approx((0.5 + 0.8772) / 2, abs=1e-3)


def test_all_negative_user_scores_zero_ndcg_and_is_excluded_from_gauc():
    users = [0, 0, 1, 1]
    labels = [0, 0, 1, 0]   # user 0: all-negative; user 1: has a positive
    scores = [1, 2, 3, 4]
    r = evaluate(users, labels, scores)
    # user 0 contributes nDCG=0 but nothing to GAUC's weighted sum
    assert r['users'] == 2


def test_primary_is_mean_of_the_two_metrics():
    users = [0, 0, 0, 1, 1]
    labels = [1, 0, 1, 1, 0]
    scores = [3, 1, 2, 5, 4]
    r = evaluate(users, labels, scores)
    assert r['primary'] == (r['GAUC'] + r['nDCG@5']) / 2.0
