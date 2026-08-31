"""[Rishi] KuaiRand-Pure data loading + official split + feature encoding.

Interface contract (frozen, CLAUDE.md §5): `load(config) -> dict[str, list[row]]`.
Nobody else re-derives or re-implements the split logic — this is the one
place it lives. `config` comes from `configs/*.yaml` via `src.config.load_config`;
never pass a raw path string in from a script body.

Only depends on the standard library and numpy.
"""
import csv
import os

import numpy as np

LABEL = 'long_view'
# 5 feature fields. This is the first place to add a feature.
FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']


def load(config):
    """Read logs + video-side features, return a dict sliced by the official split.

    config['data_dir']: directory containing the KuaiRand-Pure CSVs
    config['splits']: {'train': [lo, hi], 'valid': [lo, hi], 'test': [lo, hi]}
                       date boundaries — fixed by the task definition, do not
                       edit without full-team sign-off (CLAUDE.md §0.4).
    """
    data_dir = config['data_dir']
    splits_cfg = config['splits']

    vid2author = {}
    with open(os.path.join(data_dir, 'video_features_basic_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            vid2author[r['video_id']] = r['author_id']

    rows = []
    for f in ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv'):
        with open(os.path.join(data_dir, f)) as fh:
            for r in csv.DictReader(fh):
                rows.append((int(r['date']), r['user_id'], r['video_id'],
                             vid2author.get(r['video_id'], 'UNK'), r['tab'],
                             float(r['duration_ms']), 1 if r[LABEL] != '0' else 0))

    out = {}
    for name, (lo, hi) in splits_cfg.items():
        out[name] = [x for x in rows if lo <= x[0] <= hi]
    return out


def subsample_encoded(enc, n=5000, seed=0, extra=None):
    """Deterministic small slice of each split's *encoded* arrays, for
    --smoke-test runs only. Subsampling happens after encode(), not before:
    encode()'s vocab/dim always derive from the full train split, so a
    smoke-test checkpoint's id mapping and embedding-table size stay
    identical to a full run's — scripts/eval_checkpoint.py re-encodes from
    scratch and must land on the same ids. Never used for a reported/logged
    score.

    `extra`: optional {split_name: {key: array-like row-aligned to
    enc[split_name]}} — e.g. Vidush's IPS sample_weight (train only) or
    Nandit's sequence-id arrays (all splits). Subsampled with the exact same
    indices as enc, so these side-arrays stay row-aligned with what's
    returned. When given, returns (enc_out, extra_out) instead of just
    enc_out.
    """
    rng = np.random.default_rng(seed)
    out = {}
    out_extra = {} if extra is not None else None
    for name, (X, y, users) in enc.items():
        if len(y) <= n:
            idx = None
            out[name] = (X, y, users)
        else:
            idx = rng.choice(len(y), size=n, replace=False)
            out[name] = (X[idx], y[idx], [users[i] for i in idx])
        if extra is not None and name in extra:
            if idx is None:
                out_extra[name] = extra[name]
            else:
                out_extra[name] = {k: v[idx] for k, v in extra[name].items()}
    if extra is not None:
        return out, out_extra
    return out


def _bucket_edges(durations, n=10):
    return np.quantile(np.asarray(durations), np.linspace(0, 1, n + 1)[1:-1])


def encode(splits):
    """Map categorical features to contiguous ids. Unseen values fall into
    that field's UNK slot. Returns (X, y, users) per split — X is int32
    (N, len(FIELDS)) — plus the total embedding-table dim.
    """
    tr = splits['train']
    edges = _bucket_edges([x[5] for x in tr])

    def raw(x):
        return [x[1], x[2], x[3], x[4], str(int(np.searchsorted(edges, x[5])))]

    vocabs = [dict() for _ in FIELDS]
    for x in tr:
        for i, v in enumerate(raw(x)):
            if v not in vocabs[i]:
                vocabs[i][v] = len(vocabs[i])
    unk = [len(v) for v in vocabs]              # one UNK slot per field, at the end
    field_dims = [len(v) + 1 for v in vocabs]
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)

    enc = {}
    for name, rws in splits.items():
        X = np.empty((len(rws), len(FIELDS)), dtype=np.int32)
        y = np.empty(len(rws), dtype=np.float32)
        users = []
        for n, x in enumerate(rws):
            for i, v in enumerate(raw(x)):
                X[n, i] = vocabs[i].get(v, unk[i]) + offsets[i]
            y[n] = x[6]
            users.append(x[1])
        enc[name] = (X, y, users)
    return enc, int(sum(field_dims))
