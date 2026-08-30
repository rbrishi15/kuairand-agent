import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.features.propensity import estimate_propensity, propensity_report

# (date, user_id, video_id, author_id, tab, duration_ms, label) — src.data.load's row shape.
# 'v_common' is exposed equally by both policies (propensity ~1, weight ~1).
# 'v_over' is over-exposed by the standard policy relative to random (low weight).
# 'v_under' is under-exposed by the standard policy relative to random (high weight).
TRAIN = (
    [(20220408, 'u0', 'v_common', 'a', 't', 1000.0, 0)] * 10
    + [(20220408, 'u0', 'v_over', 'a', 't', 1000.0, 0)] * 40
    + [(20220408, 'u0', 'v_under', 'a', 't', 1000.0, 0)] * 2
)
# valid dates sit at the real train/valid boundary (20220422-20220428) so the
# dynamic max-safe-date derived from splits matches the real config's shape.
VALID = [(20220428, 'u0', 'v_common', 'a', 't', 1000.0, 0)] * 3
TEST = [(20220429, 'u0', 'v_common', 'a', 't', 1000.0, 0)] * 3
SPLITS = {'train': TRAIN, 'valid': VALID, 'test': TEST}

RANDOM_LOG = os.path.join(os.path.dirname(__file__), 'fixtures', 'random_log_fixture.csv')


def _write_random_log_fixture(extra_rows=()):
    os.makedirs(os.path.dirname(RANDOM_LOG), exist_ok=True)
    header = 'user_id,video_id,date,is_rand\n'
    rows = (
        ['u0,v_common,20220422,1'] * 10
        + ['u0,v_over,20220422,1'] * 10
        + ['u0,v_under,20220422,1'] * 40
        + list(extra_rows)
    )
    with open(RANDOM_LOG, 'w') as fh:
        fh.write(header + '\n'.join(rows) + '\n')


def _cleanup_fixture():
    if os.path.exists(RANDOM_LOG):
        os.remove(RANDOM_LOG)


def test_weights_are_ones_outside_train():
    _write_random_log_fixture()
    try:
        weights = estimate_propensity(SPLITS, RANDOM_LOG)
        assert np.all(weights['valid'] == 1.0)
        assert np.all(weights['test'] == 1.0)
    finally:
        _cleanup_fixture()


def test_train_weights_shape_and_mean_normalized():
    _write_random_log_fixture()
    try:
        weights = estimate_propensity(SPLITS, RANDOM_LOG)
        w = weights['train']
        assert w.shape == (len(TRAIN),)
        assert w.dtype == np.float32
        assert np.isclose(w.mean(), 1.0, atol=1e-4)
    finally:
        _cleanup_fixture()


def test_overexposed_video_gets_lower_weight_than_underexposed():
    _write_random_log_fixture()
    try:
        weights = estimate_propensity(SPLITS, RANDOM_LOG)
        w_by_video = {row[2]: wt for row, wt in zip(TRAIN, weights['train'])}
        assert w_by_video['v_over'] < w_by_video['v_common'] < w_by_video['v_under']
    finally:
        _cleanup_fixture()


def test_ignores_random_log_rows_past_test_window_cutoff():
    """Rows dated after the real valid split's last date (derived dynamically
    from splits, not a hardcoded constant) must not influence the estimate."""
    try:
        _write_random_log_fixture(extra_rows=['u0,v_common,20220430,1'] * 1000)
        weights_with_late_rows = estimate_propensity(SPLITS, RANDOM_LOG)

        _write_random_log_fixture()
        weights_without_late_rows = estimate_propensity(SPLITS, RANDOM_LOG)

        assert np.allclose(weights_with_late_rows['train'], weights_without_late_rows['train'])
    finally:
        _cleanup_fixture()


def test_ignores_rows_not_flagged_is_rand():
    """A row with is_rand != '1' must be excluded even if it's date-eligible —
    guards against accidentally pointing this at the wrong log file."""
    try:
        _write_random_log_fixture(extra_rows=['u0,v_under,20220422,0'] * 1000)
        weights_with_fake_rows = estimate_propensity(SPLITS, RANDOM_LOG)

        _write_random_log_fixture()
        weights_without_fake_rows = estimate_propensity(SPLITS, RANDOM_LOG)

        assert np.allclose(weights_with_fake_rows['train'], weights_without_fake_rows['train'])
    finally:
        _cleanup_fixture()


def test_alpha_and_clip_are_overridable():
    _write_random_log_fixture()
    try:
        default_weights = estimate_propensity(SPLITS, RANDOM_LOG)['train']
        tight_clip_weights = estimate_propensity(SPLITS, RANDOM_LOG, clip=(0.9, 1.1))['train']
        assert not np.allclose(default_weights, tight_clip_weights)
        # a tight [0.9, 1.1] clip must compress the spread relative to the default [0.1, 10] clip
        assert np.ptp(tight_clip_weights) < np.ptp(default_weights)
    finally:
        _cleanup_fixture()


def test_propensity_report_shape():
    _write_random_log_fixture()
    try:
        report = propensity_report(SPLITS, RANDOM_LOG)
        assert report['n_train_videos'] == 3
        assert report['n_covered_by_random_log'] == 3
        assert report['coverage_frac'] == 1.0
        assert report['weight_min'] <= report['weight_median'] <= report['weight_max']
    finally:
        _cleanup_fixture()


def test_propensity_report_flags_uncovered_videos():
    """A train video absent from the random log should show up as reduced coverage."""
    try:
        header = 'user_id,video_id,date,is_rand\n'
        rows = ['u0,v_common,20220422,1'] * 10 + ['u0,v_over,20220422,1'] * 10
        os.makedirs(os.path.dirname(RANDOM_LOG), exist_ok=True)
        with open(RANDOM_LOG, 'w') as fh:
            fh.write(header + '\n'.join(rows) + '\n')

        report = propensity_report(SPLITS, RANDOM_LOG)
        assert report['n_train_videos'] == 3
        assert report['n_covered_by_random_log'] == 2
        assert np.isclose(report['coverage_frac'], 2 / 3)
    finally:
        _cleanup_fixture()
