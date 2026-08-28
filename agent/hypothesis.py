"""Day-1 rule-based hypothesis proposer for agent/orchestrator.py.

CLAUDE.md frames the outer loop as an "autonomous research agent," which
could mean an LLM proposing and writing each iteration's code diff. On Day
1 that's overkill: nothing but the FM baseline exists yet to iterate on, so
this cycles a small deterministic FM hyperparameter/seed grid instead —
no API key, no per-iteration cost, and orchestrator.py's loop is still a
real, verifiable process rather than a stub. Once Min/Vidush/Nandit's
models land, point orchestrator.py at their configs and extend this grid
(or replace it with an LLM call) — the (index, config) -> {hypothesis,
overrides} interface stays the same either way.

Indexed by position rather than by scanning run_log.jsonl for what's
"already tried": the exact log schema (CLAUDE.md §7) has no field for
per-iteration overrides, and inventing one would put an undocumented key
in front of Sarthak's results-table parser. orchestrator.py already counts
prior iterations for this (dataset, model) to resume correctly — reusing
that count as a grid index needs nothing extra.
"""

GRID = (
    [{'seed': s} for s in range(5)]
    + [{'seed': 0, 'k': k} for k in (8, 32)]
    + [{'seed': 0, 'lr': lr} for lr in (0.0005, 0.002)]
)


def propose(index, config):
    """index: count of iterations already completed for this (dataset, model).
    Returns {'hypothesis': str, 'overrides': dict}. overrides merge into
    config['seed'] (key 'seed') or config['model'] (any other key).
    """
    overrides = GRID[index] if index < len(GRID) else {}
    desc = ', '.join(f"{k}={v}" for k, v in overrides.items()) or 'defaults'
    if index < len(GRID):
        hypothesis = (
            f"Reproduce/stress-test the FM baseline with {desc}; expect primary "
            f"within noise (std ~0.0008 across seeds) of the published baseline "
            f"if the harness is correct."
        )
    else:
        hypothesis = 'Day-1 FM grid exhausted; re-confirm the best-known config.'
    return {'hypothesis': hypothesis, 'overrides': overrides}
