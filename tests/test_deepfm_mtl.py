import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.models.deepfm_mtl import DeepFMMTL

DIM = 200
NUM_FIELDS = 5


def _synthetic_batch(n=32, seed=0):
    rng = np.random.default_rng(seed)
    X = torch.from_numpy(rng.integers(0, DIM, size=(n, NUM_FIELDS)).astype(np.int64))
    y = torch.from_numpy(rng.integers(0, 2, size=n).astype(np.float32))
    return X, y


def test_forward_shapes():
    model = DeepFMMTL(DIM, embed_dim=8, num_fields=NUM_FIELDS, aux_tasks=('click', 'like'))
    X, _ = _synthetic_batch()
    out = model(X)
    assert out['primary'].shape == (32,)
    assert set(out['aux']) == {'click', 'like'}
    assert out['aux']['click'].shape == (32,)
    assert out['sample_weight'] is None


def test_seq_embedding_hook_changes_output():
    torch.manual_seed(0)
    model = DeepFMMTL(DIM, embed_dim=8, num_fields=NUM_FIELDS, seq_dim=4)
    X, _ = _synthetic_batch()
    model.eval()
    seq = torch.randn(32, 4)
    with torch.no_grad():
        out_with = model(X, seq_embedding=seq)['primary']
        out_without = model(X, seq_embedding=torch.zeros(32, 4))['primary']
    assert not torch.allclose(out_with, out_without)


def test_sample_weight_hook_accepted_and_threaded_to_loss():
    model = DeepFMMTL(DIM, embed_dim=8, num_fields=NUM_FIELDS)
    X, y = _synthetic_batch()
    w = torch.ones(32)
    out = model(X, sample_weight=w)
    assert out['sample_weight'] is w
    loss = model.compute_loss(out, y)
    assert loss.dim() == 0 and torch.isfinite(loss)


def test_aux_loss_only_uses_heads_with_a_supplied_target():
    model = DeepFMMTL(DIM, embed_dim=8, num_fields=NUM_FIELDS, aux_tasks=('click', 'like'))
    X, y = _synthetic_batch()
    out = model(X)
    loss_primary_only = model.compute_loss(out, y)
    loss_with_click = model.compute_loss(out, y, aux_targets={'click': y})
    assert loss_primary_only.item() != loss_with_click.item()


def test_checkpoint_round_trip_matches_state_dict_contract():
    model = DeepFMMTL(DIM, embed_dim=8, num_fields=NUM_FIELDS)
    state = model.state_dict()
    reloaded = DeepFMMTL(DIM, embed_dim=8, num_fields=NUM_FIELDS)
    reloaded.load_state_dict(state)
    X, _ = _synthetic_batch()
    model.eval(); reloaded.eval()
    with torch.no_grad():
        a = model(X)['primary']
        b = reloaded(X)['primary']
    assert torch.allclose(a, b)


def test_loss_decreases_over_a_few_optimizer_steps():
    torch.manual_seed(0)
    model = DeepFMMTL(DIM, embed_dim=8, num_fields=NUM_FIELDS)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    X, y = _synthetic_batch(n=128)
    losses = []
    for _ in range(20):
        out = model(X)
        loss = model.compute_loss(out, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
