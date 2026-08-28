import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.data import FIELDS, encode, load, subsample_encoded

CONFIG = {
    'data_dir': 'data/raw',
    'splits': {
        'train': [20220408, 20220421],
        'valid': [20220422, 20220428],
        'test': [20220429, 20220508],
    },
}


def _smoke_enc():
    enc, dim = encode(load(CONFIG))
    return subsample_encoded(enc, n=2000, seed=0), dim


def test_encode_shapes_and_dtypes():
    enc, dim = _smoke_enc()
    for name in ('train', 'valid', 'test'):
        X, y, users = enc[name]
        assert X.shape[1] == len(FIELDS)
        assert X.shape[0] == len(y) == len(users)
        assert X.dtype == np.int32
        assert y.dtype == np.float32


def test_encoded_ids_stay_within_the_embedding_table():
    enc, dim = _smoke_enc()
    for name in ('train', 'valid', 'test'):
        X, _, _ = enc[name]
        assert X.min() >= 0
        assert X.max() < dim


def test_labels_are_binary():
    enc, _ = _smoke_enc()
    for name in ('train', 'valid', 'test'):
        _, y, _ = enc[name]
        assert set(np.unique(y).tolist()) <= {0.0, 1.0}
