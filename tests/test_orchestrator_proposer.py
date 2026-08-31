"""[Rishi] Tests for agent/orchestrator.py's _propose dispatch — the
--proposer grid/llm switch and its fallback behavior. The property that
matters most: an llm proposer failure (missing API key, bad response) must
fall back to the grid proposer for that iteration, never crash the loop
(CLAUDE.md §6: fail gracefully, keep going).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.orchestrator import _propose

CONFIG = {'model': {'name': 'fm'}}


def test_grid_proposer_never_errors():
    hyp, proposer_error = _propose('grid', [], CONFIG)
    assert proposer_error is None
    assert 'hypothesis' in hyp and 'overrides' in hyp


def test_llm_proposer_falls_back_to_grid_without_api_key(monkeypatch):
    monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
    hyp, proposer_error = _propose('llm', [], CONFIG)
    # falls back to the grid proposer's output, not a crash
    assert proposer_error is not None
    assert 'llm proposer failed' in proposer_error
    assert 'hypothesis' in hyp and 'overrides' in hyp


def test_llm_proposer_falls_back_on_invalid_response(monkeypatch):
    import agent.llm_hypothesis as llm_hyp

    class _BadClient:
        class messages:
            @staticmethod
            def create(**kwargs):
                raise RuntimeError('simulated API failure')

    monkeypatch.setattr(llm_hyp, 'propose_llm',
                         lambda history, config, **kw: (_ for _ in ()).throw(RuntimeError('boom')))
    hyp, proposer_error = _propose('llm', [], CONFIG)
    assert proposer_error is not None and 'boom' in proposer_error
    assert 'hypothesis' in hyp and 'overrides' in hyp
