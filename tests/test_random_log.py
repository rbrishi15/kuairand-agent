"""[Rishi] Tests for load_random_log (src/data.py) — the off-policy
validation set built from the randomized-exposure log. The critical
property: it must never include a row from the test window, since
log_random_4_22_to_5_08_pure.csv's dates span both the valid AND test
windows (CLAUDE.md §2: no test-label access during development).
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_random_log

CONFIG = {
    'data_dir': 'data/raw',
    'splits': {
        'train': [20220408, 20220421],
        'valid': [20220422, 20220428],
        'test': [20220429, 20220508],
    },
}


def test_never_reads_test_window_dates():
    rows = load_random_log(CONFIG)
    test_lo, test_hi = CONFIG['splits']['valid'][1] + 1, 20220508
    assert all(row[0] < test_lo or row[0] > test_hi for row in rows)


def test_only_within_the_valid_window():
    lo, hi = CONFIG['splits']['valid']
    rows = load_random_log(CONFIG)
    assert len(rows) > 0
    assert all(lo <= row[0] <= hi for row in rows)


def test_only_is_rand_rows_included():
    # cross-check against a raw scan of the file
    lo, hi = CONFIG['splits']['valid']
    path = os.path.join(CONFIG['data_dir'], 'log_random_4_22_to_5_08_pure.csv')
    expected = 0
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r.get('is_rand') == '1' and lo <= int(r['date']) <= hi:
                expected += 1
    assert len(load_random_log(CONFIG)) == expected


def test_row_shape_matches_load():
    rows = load_random_log(CONFIG)
    row = rows[0]
    assert len(row) == 7  # (date, user_id, video_id, author_id, tab, duration_ms, label)
    assert isinstance(row[0], int)
    assert row[6] in (0, 1)
