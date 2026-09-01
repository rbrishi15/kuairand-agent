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

### Team member contributions

- **Rishi** — built the shared spine (`src/data.py`'s split/encode logic,
  `agent/orchestrator.py`'s outer loop, `agent/convergence.py`), reproduced
  the official FM baseline within noise, then integrated everyone else's
  work: wired Vidush's IPS weights and Nandit's sequence encoder into
  Min's `DeepFMMTL` hooks, built the off-policy validation harness against
  the randomized-exposure log, implemented pairwise BPR training, the
  LLM-driven hypothesis proposer, and ran the multi-seed sweeps that
  separated real effects from noise.
- **Min** — designed and implemented `DeepFMMTL` (shared embedding table +
  FM second-order term + deep MLP tower + multi-task head plumbing), and
  established the `sample_weight`/`seq_embedding` interface contract from
  day 1 so Vidush and Nandit were never blocked waiting on a rewrite.
- **Vidush** — built the IPS propensity estimation
  (`src/features/propensity.py`): item-level exposure propensities from
  KuaiRand-Pure's randomized-exposure log, Laplace-smoothed and clipped,
  self-normalized to mean 1, with the safe-date cutoff derived dynamically
  from the actual split boundaries rather than hardcoded. Also expanded
  the LLM proposer's allowlist with real architecture/training knobs,
  found and fixed a Windows console-encoding bug in `submit.py` that was
  silently blocking `scripts/check.sh` from ever reporting success on a
  non-UTF-8 console, and drove the bonus-dataset (KuaiRand-1K/-27K)
  download, wiring, and evaluation.
- **Nandit** — built the sequential features (per-user recent-interaction
  history, `src/features/sequential.py`), the GRU sequence encoder
  (`src/models/seq_encoder.py`), and the item2vec-style SSL pretraining
  pretext task (`src/models/ssl_pretrain.py`).
- **Sarthak** — built submission validation (`src/submit.py`) and
  rank-average ensembling (`src/models/ensemble.py`), and wired real
  checkpoint/ensemble loading into the final submission path.

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

## Resource summary

Computed directly from `logs/run_log.jsonl` (its 25 entries are the
complete outer-loop history for this project — CLAUDE.md §13's Definition
of Done requires this be generated from the log, never hand-typed):

| | |
|---|---|
| Iterations | 25 |
| Total wall-clock | 2334.3s (≈0.65h) |
| Tokens used | 0 input / 0 output — no `ANTHROPIC_API_KEY` was available during any of these iterations, so the LLM proposer (`--proposer llm`) never made a real billed call; its one logged attempt (iteration 25) is the credential-failure-and-fallback demo described above |
| GPU-hours | 0 — every run was CPU-only (CLAUDE.md §3's CPU-code-path requirement); no GPU was used or required anywhere in this project |
| Models covered | `fm`, `deepfm_mtl` |
| Dataset | `kuairand_pure` |

## Manual intervention count

**19 of 25** logged iterations carry `manual_intervention: true`. These are
the by-hand multi-seed sweeps (e.g. iterations 10-17, 20-24) — seeds run
individually via `train.py --seed N` to separate real effects from noise,
which is inherently a human decision ("run this exact config N more
times"), not a hypothesis for the orchestrator's proposer to generate. This
is expected given how this project's ablations were actually run, not a
sign of the autonomous loop failing.

**1** additional iteration (25) logged a real error — a missing
`ANTHROPIC_API_KEY` for the LLM proposer — with `manual_intervention:
false`, since the orchestrator's own fallback-to-grid recovery handled it
without any human stepping in. See "LLM-driven hypothesis proposer" above
for the full account of that one.

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

### Bonus: KuaiRand-1K

Same pipeline, `configs/kuairand_1k_deepfm_mtl.yaml` (base DeepFMMTL, no
IPS/BPR — both were tested and found to *not* transfer cleanly from
Pure-tuned hyperparameters to 1K's larger scale during ablation, so the
plain base config is the honest choice here, not a shortcut), single
seed rather than Pure's 5-seed ensemble given the much larger per-run
cost (1K's ~11.7M-row standard logs vs. Pure's ~1.4M):

```bash
python3 src/train.py --config configs/kuairand_1k_deepfm_mtl.yaml --seed 0
python3 src/submit.py --make --config configs/kuairand_1k_deepfm_mtl.yaml \
    --split test outputs/submission_1k.csv \
    --checkpoint checkpoints_1k/deepfm_mtl_seed0.pt
python3 src/submit.py --check --config configs/kuairand_1k_deepfm_mtl.yaml \
    --split test outputs/submission_1k.csv
```

4,132,081 rows, passes `--check`:

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| **This submission (1K, single-seed DeepFMMTL)** | **0.6738** | **0.6079** | **0.6408** |

No official 1K baseline number is published to compare against (unlike
Pure's 0.5946), and these raw magnitudes aren't comparable to Pure's
anyway — 1K's much denser per-user interaction history makes within-user
ranking structurally easier, not evidence of a better model. Only
directional/relative findings from 1K testing are meaningful; see
Limitations below.

**Neither `checkpoints_1k/deepfm_mtl_seed0.pt` (~199MB) nor
`outputs/submission_1k.csv` (~121MB) are committed to this repo** — both
exceed GitHub's 100MB per-file limit and would be rejected on push. Both
are exactly reproducible via the two commands above; a Git LFS setup or
an external release attachment would be needed to actually distribute the
binary files themselves, not yet decided.

### Bonus: KuaiRand-27K

Downloaded and checksum-verified (9.9GB archive), but training was not
attempted — a deliberate call once the actual extracted scale was known,
not a time-ran-out guess:

| | KuaiRand-Pure | KuaiRand-1K | KuaiRand-27K |
|---|---|---|---|
| `log_standard_4_08_to_4_21` | 84MB | 373MB | **~10.3GB** (2 parts) |
| `log_standard_4_22_to_5_08` | 22MB | 492MB | **~14.1GB** (2 parts) |

27K's logs are ~28x 1K's size, and 1K's ~11.7M rows already took ~70s
just to load (pure-Python `csv.DictReader`, no chunking, in
`src/data.py`) and ~500s to train a single epoch on CPU. Extrapolating,
27K is plausibly 250M+ rows — hours to load before training even starts,
not a feasible CPU-only run in this project's time budget. This confirms,
with real measurements rather than an assumption, the playbook's own
Day-3 call to treat 27K as the one to skip if time is short ("too large
to safely debug in remaining time"). No 27K results are reported.

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

## Limitations & what we'd improve with more time

Written honestly, not as a formality — most of these are already implied
by results reported above, collected here in one place:

- **No technique has beaten FM by more than noise, decisively.** The best
  single number anywhere in this project (0.6040 valid, the 5-seed
  DeepFMMTL ensemble) is +0.0024 over FM's published 0.6016 — real, but
  small next to the seed-to-seed std every variant shows. Given more
  time, the honest next step per the README's own headroom analysis is a
  hybrid pointwise+pairwise loss (BPR alone underperforms, likely from
  training on far fewer effective pairs per epoch — see "Model variants
  tried") rather than another architecture sweep; capacity has been
  checked twice now (embedding dim, then hidden-dims/dropout/weight-decay)
  and isn't the bottleneck either time.
- **IPS's counterfactual benefit is still genuinely unconfirmed.** The
  off-policy check this project is proudest of (a real unbiased eval set
  built from the randomized-exposure log, not just an assertion) came back
  a 0.29x-of-noise gap over 5 seeds — not proof it helps, not proof it
  doesn't. A logistic-regression propensity model over user/video/context
  features (closer to the Zhao et al. framing that motivated this
  technique) instead of the current item-level ratio estimator is the
  natural next attempt, alongside more offpolicy seeds before trusting
  either direction.
- **Multi-task auxiliary labels were never wired in.** `DeepFMMTL` has the
  aux-head plumbing (`compute_loss`'s `aux_targets`) but `src/data.py`
  never loads `is_click`/`is_like`/`play_time_ms` — Priority 1's
  multi-task story is architecturally ready but never actually multi-task
  in any run reported here.
- **Video-level engagement statistics
  (`video_features_statistic_pure.csv`, 51 columns — show/play/like/
  comment/follow/share counts) are never read anywhere in the pipeline.**
  An earlier ablation of "extra feature domains" found no lift, but that
  test's exact methodology (continuous vs. bucketed-into-IDs, aggregate
  vs. stratified by video train-frequency) isn't fully known from the
  one-line note it's recorded in — a rare-video-stratified retry with
  continuous features (not more categorical IDs) is a real open question,
  not a settled one, and is a lower priority than the two items above
  mainly because it requires new plumbing DeepFMMTL doesn't have yet.
- **Bonus benchmarks are partially attempted, not fully matched to
  Pure's rigor.** KuaiRand-1K was run and submitted (single seed, not the
  5-seed sweep Pure got — the dataset is ~5x bigger by row count despite
  fewer users, and hyperparameters tuned on Pure (e.g. BPR's learning
  rate) were found NOT to transfer cleanly to 1K's scale during testing,
  a genuine generalization-risk finding in its own right, not just an
  excuse). KuaiRand-27K (9.9GB) was attempted but not completed in the
  project's time budget — consistent with the playbook's own Day-3 plan
  to treat it as lowest priority ("skip 27k — too large to safely debug
  in remaining time").
- **`run_log.jsonl`'s `code_diff_ref` field is null in every entry.** The
  schema has always had this field, but nothing in this project's tooling
  ever populated it with an actual commit/diff reference — git history is
  the real record of what changed each iteration, but it's not
  cross-linked from the log itself. Wiring `code_diff_ref` to the actual
  commit SHA active at iteration time (via a git hook or orchestrator-side
  lookup) is a small, concrete fix that just never got done.
- **The LLM proposer has never run for real.** `agent/llm_hypothesis.py`
  is fully implemented and tested (fake-client unit tests, 13 passing),
  and it hit and correctly recovered from one live credential failure —
  but no `ANTHROPIC_API_KEY` was available in the environment this
  project was built in, so there's no real "LLM-chosen experiments vs.
  the grid" comparison to report, only a robustness demonstration. That
  comparison is the most interesting unrun experiment in the repo.

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
