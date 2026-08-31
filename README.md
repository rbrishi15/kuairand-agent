# kuairand-agent

Autonomous ML research agent for the KuaiRand-Pure recommendation benchmark
— TikTok TechJam.

Full spec: [`CLAUDE.md`](CLAUDE.md) (machine-actionable, source of truth for
anything code/process-related) and `KuaiRand_Team_Playbook.docx`
(human-facing rationale). Read `CLAUDE.md` before touching any code.

## Team

| Branch | Human | Owns | Technique |
|---|---|---|---|
| `feat/harness` | Rishi | `agent/orchestrator.py`, `agent/convergence.py`, `src/data.py`, `logs/` | Baseline reproduction, integration |
| `feat/multitask` | Min | `src/models/deepfm_mtl.py`, `src/features/base.py` | Multi-task DeepFM |
| `feat/ips-debias` | Vidush | `src/features/propensity.py` | IPS debiasing |
| `feat/sequential-ssl` | Nandit | `src/models/seq_encoder.py`, `src/features/sequential.py`, `src/models/ssl_pretrain.py` | Sequential features + SSL |
| `feat/eval-submission` | Sarthak | `src/submit.py`, `src/models/ensemble.py`, results table | Ensembling |

Only edit files your role owns unless told otherwise. `main` is merge-only,
reviewed by Rishi.

## Setup

```bash
# pinned Python version — see .python-version
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# data (same source the starter kit uses; not committed — see .gitignore)
curl -L -o /tmp/KuaiRand-Pure.tar.gz https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar xzf /tmp/KuaiRand-Pure.tar.gz -C /tmp
mkdir -p data/raw && cp /tmp/KuaiRand-Pure/data/*.csv data/raw/
```

Or use the `Dockerfile` — the environment tiebreaker per `CLAUDE.md` §3: if
something behaves differently on two machines, the question is whether it
behaves correctly inside the container, not whether the code is wrong.

## Running

```bash
# one training run
python src/train.py --config configs/kuairand_pure.yaml

# the outer agent loop — reads logs/run_log.jsonl, proposes the next
# experiment, trains, evaluates, logs, checks convergence, repeats
python agent/orchestrator.py --config configs/kuairand_pure.yaml

# generate + validate a submission
python src/submit.py --make  --config configs/kuairand_pure.yaml --split test outputs/submission.csv
python src/submit.py --check --config configs/kuairand_pure.yaml --split test outputs/submission.csv
```

Before every push:

```bash
bash scripts/check.sh
```

## Task definition (fixed — do not change)

| | |
|---|---|
| Task | **within-user ranking** — rank each user's own logged impressions, not full-catalog retrieval |
| Label | `long_view` (native column, 0/1) |
| Metrics | `GAUC`, `nDCG@5`; **primary = mean of both** |
| Split | train `20220408–20220421` / valid `20220422–20220428` / test `20220429–20220508` |
| Zero-positive users | nDCG scored 0.0, included in the average; excluded from GAUC |
| nDCG gain | `2^rel − 1` |

Implementation: `src/evaluate.py` — starter-kit provided, frozen, do not
modify. `scripts/eval_checkpoint.py` wraps it with a checkpoint-loading CLI
without touching the scoring logic itself.

## Baseline ladder (test set)

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| random (sanity check) | 0.4996 | 0.4511 | 0.4753 |
| item popularity (trivial) | 0.6308 | 0.5121 | 0.5715 |
| **FM (official baseline — beat this)** | **0.6610** | **0.5282** | **0.5946** |
| oracle ceiling (perfect ranking) | 1.0000 | 0.7289 | 0.8645 |

nDCG's ceiling is 0.7289, not 1.0 — 27.1% of test users have no positive
label at all (nDCG pinned at 0, unfixable by any model). **Judge progress
against 0.8645, not 1.0.** Full detail in `baseline_scores.json`.

### Already tried, no lift — don't repeat

- Adding CWM's 13 feature domains vs. the current 5: no measurable
  difference (noise-level).
- Embedding dim k = 8 / 16 / 32: barely moves the needle.
- `user_id × video_id` already absorbs most learnable signal from static
  features; a user-side feature's first-order term contributes exactly 0
  to within-user ranking (it's constant within a user's own group).

### Where headroom likely is, in priority order

1. Multi-task auxiliary labels (`is_click`, `is_like`, `play_time_ms`, ...) —
   `DeepFMMTL` has the aux-head plumbing (`compute_loss`'s `aux_targets`)
   but `src/data.py` doesn't expose those columns yet.
2. Counterfactual watch-time modeling — IPS reweighting (below) is a step
   toward this, not the full picture.
3. A hybrid pointwise + pairwise loss — BPR alone (below) underperforms
   plain BCE, most likely because it trains on far fewer effective
   examples per epoch; adding the pairwise term on top of BCE rather than
   replacing it is the natural next thing to try, not yet attempted.

Model capacity, user interaction sequences, IPS debiasing, and pairwise
ranking loss have now actually been tried (see below) — see the results
before assuming there's easy lift left in any of them.

## Model variants tried

`configs/kuairand_pure_deepfm_mtl*.yaml` — DeepFMMTL (Min, Priority 1) with
Vidush's IPS weights and/or Nandit's sequence encoder optionally wired in
via two model-config flags (`use_ips`, `use_seq`; both default false, see
`src/train.py`'s module docstring for the full option list; `--seed`
overrides `config['seed']` for sweeping). Real (non-smoke) runs on
KuaiRand-Pure, multiple seeds each — mean ± population std:

| Variant | Config | valid primary (n seeds) |
|---|---|---|
| FM (official baseline, published) | `kuairand_pure.yaml` | 0.6016 (std 0.0008, n=5) |
| DeepFMMTL | `kuairand_pure_deepfm_mtl.yaml` | **0.6033 ± 0.0003** (n=5) |
| DeepFMMTL + IPS | `..._ips.yaml` | 0.6012 ± 0.0003 (n=5) |
| DeepFMMTL + sequences | `..._seq.yaml` | 0.6035 ± 0.0001 (n=2) |
| DeepFMMTL + IPS + sequences | `..._full.yaml` | 0.6018 ± 0.0001 (n=2) |
| DeepFMMTL + pairwise BPR | `..._bpr.yaml` | 0.6018 ± 0.0003 (n=5) |

With enough seeds, this actually separates from noise: **base vs. +IPS is a
real, reproducible ~0.002 gap** (~6× the combined std) — IPS is not just
flat-line noise, it measurably *costs* `valid primary`. See below for why
that's not necessarily bad news. +sequences (n=2 so far) looks flat-to-
slightly-better than base; +IPS+sequences lands almost exactly at the
FM baseline. None of this is a confirmed win over plain DeepFMMTL yet —
sequences and the combined variant still need 3+ more seeds each before
trusting the (very small, n=2) gaps between them. Three non-obvious things
worth knowing before extending this ablation:

- **Pairwise BPR (`run_deepfm_mtl_bpr` in `src/models/deepfm_mtl.py`)
  underperforms plain pointwise BCE, and it's not just noise** (~4.4× the
  combined std over 5 seeds) — despite directly optimizing per-user
  pairwise ordering, the same thing GAUC measures. It even loses on GAUC
  itself, not just primary. Root cause, not yet fixed: BPR only trains on
  `(pos, neg)` pairs, and `long_view=1` is the minority class here — its
  default `pairs_per_epoch` (one pass over eligible positives) sees only
  ~383K pairs/epoch versus pointwise's 1.14M rows, roughly a third of the
  effective training signal per epoch, even though 92.7% of users are
  individually eligible (have both classes). A hybrid loss (BCE + a BPR
  term on top, not instead) is the natural next attempt — theoretical
  alignment with the metric isn't enough on its own if the loss sees much
  less data per step.

- **IPS looking flat-to-slightly-worse on `valid` doesn't necessarily mean
  it's not working** — `valid` is drawn from the standard (biased) log, not
  the random-exposure one, so it structurally can't reward what IPS is
  trying to fix. **This was checked, not just asserted** (see off-policy
  validation below) — and the check did not confirm it either. Take "IPS
  might help off-policy" as an open question, not a settled one.
- **The sequence encoder is the slow one** — a real (non-smoke) run takes
  ~50s/epoch vs ~1s/epoch for plain DeepFMMTL, all in the GRU forward/backward
  over padded sequences on CPU. Budget for that before running the seq or
  full ablations again.

## Off-policy validation (playbook idea #7)

`src/data.py`'s `load_random_log(config)` builds a genuinely unbiased
evaluation set from `log_random_4_22_to_5_08_pure.csv` (uniformly-random
exposure, not the production policy) — restricted to the official valid
window's dates only, since that file's date range also spans the test
window and reading labels from there would be exactly the test-label
access CLAUDE.md §2 forbids. `scripts/eval_checkpoint.py --split
offpolicy` scores any checkpoint against it. The motivating question: does
IPS's cost on the (still-biased) `valid` split actually pay off once
measured on exposure that isn't biased?

**Answer, from all available seeds — no, not detectably:**

| Variant | offpolicy primary (n seeds) |
|---|---|
| DeepFMMTL | 0.3702 ± 0.0006 (n=5) |
| DeepFMMTL + IPS | 0.3704 ± 0.0007 (n=5) |
| DeepFMMTL + BPR | 0.3709 ± 0.0011 (n=5) |
| DeepFMMTL + sequences | 0.3736 ± 0.0005 (n=2) |
| DeepFMMTL + IPS + sequences | 0.3717 ± 0.0013 (n=2) |

Base vs. +IPS is a 0.0002 gap — **0.29× the combined std, not distinguishable
from noise.** Worth stating plainly since it contradicts a plausible-sounding
hypothesis: looking only at seed 0 (base 0.3699 vs. +IPS 0.3711) looked like
a real win for IPS, the same mistake flagged earlier in this README for the
`valid`-side ablation. It wasn't, once all 5 seeds were checked. IPS's
counterfactual benefit remains genuinely unconfirmed either way — not
proven to help, not proven not to. +sequences again shows the best mean
(consistent with its `valid` result) but is still only n=2.

Absolute magnitudes here (~0.37) aren't comparable to `valid`'s (~0.60) —
different label base rate under random exposure — only the *relative*
ordering between variants, evaluated on the same offpolicy set, is
meaningful.

## LLM-driven hypothesis proposer (`--proposer llm`)

`agent/orchestrator.py --proposer llm` uses `agent/llm_hypothesis.py`
instead of the deterministic grid: a Claude API call reasons over the real
run history and picks the next experiment. The safety boundary is
deliberate and narrow — the LLM **chooses a value for a fixed, validated
set of existing config keys** (`ALLOWED_OVERRIDES` in that file); it never
writes or executes code, never sees a file path, never touches
`src/data.py`'s split logic or `src/evaluate.py`. This is the scaled-down,
actually-safe slice of the "AIDE-style" autonomous-agent idea — real
reasoning over real infrastructure, without the risk surface of an LLM
generating and running arbitrary training code (a much bigger, riskier
project not attempted here).

**Honest status: built and tested, not yet demonstrable end-to-end.** No
`ANTHROPIC_API_KEY` was available in the environment this was built in, so
there's no real comparison of "LLM-chosen experiments vs. the grid" to
report — that requires a key someone actually has. What **is** verified:

- `tests/test_llm_hypothesis.py` (8 tests, fake client, no API key needed)
  — a malformed response, a disallowed key, a wrong type, or an
  out-of-enum value are all rejected, never silently accepted.
- `tests/test_orchestrator_proposer.py` — the orchestrator falls back to
  the grid proposer on any proposer failure rather than halting.
- **A real, live run** (`python agent/orchestrator.py --config
  configs/kuairand_pure_deepfm_mtl.yaml --proposer llm`, iteration 25 in
  `logs/run_log.jsonl`) actually hit the missing-credentials error live,
  logged the real exception text, fell back to the grid proposer, and
  completed the iteration normally — `manual_intervention: false`,
  correctly, since nothing about that recovery needed a human. That's a
  genuine (if modest) demonstration of the Robustness criterion CLAUDE.md
  §6 cares about: not "never fails," but "fails gracefully and keeps
  going," including for the proposer itself, not just training.

Once a key is available: rerun the same command and diff what the LLM
actually proposes against `agent/hypothesis.py`'s grid.

## Ensembling (Priority 5)

`src/models/ensemble.py`'s `ensemble_predict()` blends any mix of
checkpoints via **rank-average**: each checkpoint's raw score is converted
to a within-user percentile rank before blending (`scripts/eval_ensemble.py`
is the CLI), since FM's and DeepFMMTL's raw scores aren't on comparable
scales and only within-user ordering is ever measured. Weight per
checkpoint, not per technique, by default — pass explicit `--weight` flags
to give each technique equal say regardless of how many seeds it has.

**Result, and the important caveat that comes with it:**

| Ensemble | valid primary |
|---|---|
| DeepFMMTL, 5 seeds averaged (no other technique) | **0.6040** |
| DeepFMMTL + sequences, 2 seeds averaged (no other technique) | 0.6038 |
| DeepFMMTL (5 seeds) + sequences (2 seeds), 50/50 by technique | 0.6040 |
| All 5 techniques (19 checkpoints), weighted equally per technique | 0.6040 |

This beats every individual single-seed number reported anywhere above
(best single run was 0.6040 too, coincidentally, but the *mean* single-seed
numbers were 0.6032-0.6035). **The gain is pure variance reduction from
averaging multiple seeds, not complementary diversity between
techniques** — base-seeds-alone already hits the same 0.6040 that
combining base+sequences does, and throwing in IPS/BPR/full on top changes
nothing further. Don't oversell this as "ensembling combined diverse
signals" — the honest story is "training the same model 5 times and
averaging is worth about +0.0007 over any single run," which is a real,
legitimate, low-risk thing to ship, just not the complementary-diversity
story that would have been more interesting to report.

## Final submission

`outputs/submission.csv` — generated via `src/submit.py`'s `--checkpoint`
path (added on top of the ensembling work above; previously `--make` could
only retrain a fresh FM, with no way to point it at a real checkpoint):

```bash
python3 src/submit.py --make --config configs/kuairand_pure_deepfm_mtl.yaml \
    --split test outputs/submission.csv \
    --checkpoint checkpoints/deepfm_mtl_seed0.pt \
    --checkpoint checkpoints/deepfm_mtl_seed1.pt \
    --checkpoint checkpoints/deepfm_mtl_seed2.pt \
    --checkpoint checkpoints/deepfm_mtl_seed3.pt \
    --checkpoint checkpoints/deepfm_mtl_seed4.pt
python3 src/submit.py --check --config configs/kuairand_pure_deepfm_mtl.yaml \
    --split test outputs/submission.csv
```

Uses the 5-seed plain-DeepFMMTL ensemble, not the full 19-checkpoint blend
above — per the ensembling section's own finding, the larger blend doesn't
score any higher on `valid` (0.6040 either way), so the simpler 5-checkpoint
version is the honest choice, not a shortcut.

170,588 rows, passes `--check` (header, row count, `row_id` continuity,
`(user_id, video_id)` alignment, no NaN/Inf). Scored against the real
(previously untouched) test-set labels only for this one final reporting
pass, per CLAUDE.md §2 — not used anywhere during model selection above:

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| FM (official baseline, published) | 0.6610 | 0.5282 | 0.5946 |
| **This submission (5-seed DeepFMMTL ensemble)** | **0.6656** | **0.5310** | **0.5983** |

+0.0037 primary over the official baseline on held-out test — in the same
ballpark as (here, slightly larger than) the `valid`-side gap for this same
5-seed ensemble (0.6040 vs. FM's published 0.6016, +0.0024), consistent
with the rest of this README's theme: real but modest, not a headline win,
and reported once rather than being used to pick among checkpoints after
the fact.

## Submission format

CSV, header + one row per evaluation-set row:

```
row_id,user_id,video_id,score
0,0,7531,-3.34176
```

`row_id` is the join key — `(user_id, video_id)` repeats in the eval set
(3.06% of test rows), so it can't be a primary key. `src/submit.py --check`
rejects header mismatches, row-count mismatches, `row_id` gaps, misaligned
`(user_id, video_id)`, and non-numeric/NaN/Inf scores.

## Repo layout

See `CLAUDE.md` §4 for the authoritative tree and §5 for the interface
contracts between roles (most important one: `DeepFMMTL.forward(x,
sample_weight=None, seq_embedding=None)` — both hooks exist from day 1 in
`src/models/deepfm_mtl.py` so Vidush and Nandit are never blocked on a
rewrite).

## CI

`.github/workflows/check.yml` runs `scripts/check.sh` on GitHub's neutral
machine for every PR — required to pass before review. If it's green on
Actions but red on your laptop, that's a local environment issue (check
`.python-version` and `requirements.txt` pins first), not a code bug.
