"""Day-1 rule-based hypothesis proposer for agent/orchestrator.py.

CLAUDE.md frames the outer loop as an "autonomous research agent," which
could mean an LLM proposing and writing each iteration's code diff. On Day
1 that's overkill: nothing but the FM baseline exists yet to iterate on, so
this cycles a small deterministic grid instead — no API key, no
per-iteration cost, and orchestrator.py's loop is still a real, verifiable
process rather than a stub. If a future iteration wants an LLM proposing
hypotheses instead, the (index, config) -> {hypothesis, overrides}
interface stays the same either way.

Indexed by position rather than by scanning run_log.jsonl for what's
"already tried": the exact log schema (CLAUDE.md §7) has no field for
per-iteration overrides, and inventing one would put an undocumented key
in front of Sarthak's results-table parser. orchestrator.py already counts
prior iterations for this (dataset, model) to resume correctly — reusing
that count as a grid index needs nothing extra.

One grid per model name, since override keys are model-specific (FM's `k`
means nothing to DeepFMMTL, which uses `embed_dim`). The deepfm_mtl grid
below mirrors the ablation Rishi ran by hand once Min/Vidush/Nandit's
pieces merged: base, +IPS (Vidush), +sequences (Nandit), +both — so a
future orchestrator run against configs/kuairand_pure_deepfm_mtl*.yaml
resumes that exploration in a properly logged, resumable way instead of
silently repeating FM-only knobs.
"""

FM_GRID = (
    [{'seed': s} for s in range(5)]
    + [{'seed': 0, 'k': k} for k in (8, 32)]
    + [{'seed': 0, 'lr': lr} for lr in (0.0005, 0.002)]
)

DEEPFM_GRID = (
    [{'seed': s} for s in range(3)]
    + [{'seed': 0, 'use_ips': True}]
    + [{'seed': 0, 'use_seq': True}]
    + [{'seed': 0, 'use_ips': True, 'use_seq': True}]
)
# Note: Rishi later ran a fuller multi-seed sweep by hand (5 seeds each for
# base/+IPS, 2 for +seq/+both — logs/run_log.jsonl iterations 10-19-ish) to
# tell noise from signal, since n=1 per variant above was barely past FM's
# own documented std. That sweep is *not* reflected in this grid (index-based
# proposing can't retroactively reconcile ad-hoc runs without an
# undocumented log field — see the module docstring). A future orchestrator
# run against deepfm_mtl.yaml will hit "grid exhausted" immediately, which
# is safe but uninteresting; extend this grid with genuinely new ideas
# (more seq/full seeds, new hyperparameters) rather than expecting it to
# resume where the manual sweep left off.

GRID_BY_MODEL = {'fm': FM_GRID, 'deepfm_mtl': DEEPFM_GRID}


def propose(index, config):
    """index: count of iterations already completed for this (dataset, model).
    Returns {'hypothesis': str, 'overrides': dict}. overrides merge into
    config['seed'] (key 'seed') or config['model'] (any other key).
    """
    model_name = config['model']['name']
    grid = GRID_BY_MODEL.get(model_name, ())
    overrides = grid[index] if index < len(grid) else {}
    desc = ', '.join(f"{k}={v}" for k, v in overrides.items()) or 'defaults'
    if index < len(grid):
        if model_name == 'fm':
            hypothesis = (
                f"Reproduce/stress-test the FM baseline with {desc}; expect primary "
                f"within noise (std ~0.0008 across seeds) of the published baseline "
                f"if the harness is correct."
            )
        else:
            hypothesis = (
                f"Train {model_name} with {desc}; IPS is expected to look flat-to-"
                f"slightly-worse on this (still-biased) validation split even if it "
                f"helps counterfactually, since valid isn't drawn from the random-"
                f"exposure log — sequences are the more likely source of real lift."
            )
    else:
        hypothesis = f"{model_name} grid exhausted; re-confirm the best-known config."
    return {'hypothesis': hypothesis, 'overrides': overrides}
