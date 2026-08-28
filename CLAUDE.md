# CLAUDE.md — KuaiRand Autonomous ML Research Agent (Team Edition)

This is the single source of truth for any Claude instance working on this
repo — whether that's Claude Code reading this file automatically from the
repo root, or Claude Pro with this file uploaded directly into the
conversation. All five team members use this exact same file. Do not
maintain separate per-person copies — if something needs to change, it
changes here, once, and everyone re-syncs.

Companion document: `TEAM_PLAYBOOK.docx` — human-facing narrative version
of the roles and rationale below. This file is the machine-actionable
version. If the two ever disagree, this file wins for anything code- or
process-related; the playbook wins for anything about intent or "why."

---

## 0. Read this first — how to use this file

1. **Identify which role you're operating as** before touching any code.
   Check, in order: (a) the current git branch name against the table in
   §1, (b) if ambiguous, ask the human operator directly — "which of the
   five roles are we working as in this session?" Do not guess.
2. **Only edit files owned by that role** (§1 table) unless the human
   operator explicitly says otherwise in this session. If a task requires
   touching another role's file, stop and say so — don't silently do it.
3. **Run `scripts/check.sh` before considering any change complete.** A
   change that hasn't passed this script is not done, regardless of
   whether the code "looks right."
4. **Never modify `src/data.py`'s split logic, `configs/*.yaml` seeds, or
   `agent/convergence.py`'s thresholds** unless you are specifically
   operating as Rishi's role and the human operator has explicitly asked
   for that change. These are shared contracts — silently changing them
   invalidates every other branch's results.

---

## 1. Roles, ownership, and role detection

| Branch name | Human | Role | Owns (only edit these without asking) | Technique |
|---|---|---|---|---|
| `feat/harness` | Rishi | Project lead + integration owner | `agent/orchestrator.py`, `agent/convergence.py`, `src/data.py`, `logs/`, `configs/*.yaml` (seed/split values) | Baseline reproduction |
| `feat/multitask` | Min | Model lead | `src/models/deepfm_mtl.py`, `src/features/base.py` | Priority 1 — multi-task DeepFM |
| `feat/ips-debias` | Vidush | Debiasing lead | `src/features/propensity.py` | Priority 2 — IPS debiasing |
| `feat/sequential-ssl` | Nandit | Sequence/SSL lead | `src/models/seq_encoder.py`, `src/features/sequential.py`, `src/models/ssl_pretrain.py` | Priority 3 & 4 — sequential features + SSL |
| `feat/eval-submission` | Sarthak | Eval & deliverables lead | `src/submit.py`, `src/models/ensemble.py`, `README.md`, results table | Priority 5 — ensembling |
| `main` | — | Integration branch | Nobody edits directly — merges only, via PR, reviewed by Rishi | — |

If working on `main` or an unrecognized branch, do not assume a role or
technique — ask the human operator what the current task is before writing
code.

---

## 2. Hard constraints — never violate these, regardless of role

- **No external training data.** Only the KuaiRand CSVs in `data/raw/` may
  train anything. No augmenting, joining, or scraping additional data.
- **No hidden-test-label access during development.** Never read, print,
  or compute statistics from the `test` split's labels. Only train +
  validation are visible until the single final scoring pass.
- **No pretrained weights trained on this benchmark's test labels.**
  Pretrained weights are otherwise allowed freely — the restriction is
  specifically about prior exposure to these test answers, not pretrained
  weights in general. If unsure whether a checkpoint qualifies, don't use
  it and flag it to the human operator.
- **Compute budget per benchmark run:** 50 iterations (hard cap) OR
  convergence (validation primary score improves by < ε=0.002 over the
  last N=3 consecutive iterations) OR 6h wall-clock — whichever triggers
  first.
- **Every iteration must be logged** (schema in §7) before starting the
  next one. An unlogged iteration doesn't count as having happened, and
  breaks Sarthak's downstream results table.
- **No absolute file paths anywhere in committed code.** Every path goes
  through `configs/*.yaml`. A hardcoded path is treated as a bug, not a
  style issue — it silently breaks the pipeline on someone else's machine.
- **No direct pushes to `main`.** All work happens on the role's own
  branch, merged via PR, reviewed by Rishi.

---

## 3. Cross-platform / environment rules

The five of you are on different OSes and possibly different hardware.
These rules exist specifically to prevent "works on my machine" failures —
follow them exactly, don't approximate.

- **Python version:** whatever is pinned in `.python-version`. If your
  local interpreter doesn't match, that's the bug to fix before writing
  any code — don't code around a version mismatch.
- **Dependencies:** exact pins in `requirements.txt` (e.g. `torch==2.x.x`,
  never a range). If a package you need isn't pinned yet, add it with an
  exact version and flag the addition — don't install unpinned packages
  ad hoc.
- **Docker is the tiebreaker.** If code behaves differently on two
  people's machines, the question to answer is "does it behave correctly
  inside the repo's `Dockerfile`?" If yes, the bug is local environment,
  not code — don't spend time debugging the code further.
- **Every model must have a CPU code path.** GPU-only code that only
  works on one person's hardware cannot be verified or merged by anyone
  else. GPU acceleration is fine as an optional speedup, not a
  requirement.
- **Random seeds are fixed centrally** in `configs/*.yaml` — never
  re-randomize inside a script. A result that can't be reproduced by a
  teammate running the same config is not a valid result.
- **Timestamps in `run_log.jsonl` are UTC**, always — use
  `datetime.now(timezone.utc)`, never local time.

---

## 4. Repo layout

```
kuairand-agent/
├── .python-version            # pinned Python version — do not change without team agreement
├── requirements.txt           # exact-pinned dependencies
├── Dockerfile                 # environment ground truth
├── .github/workflows/check.yml  # required CI — runs scripts/check.sh on every PR
├── data/
│   ├── raw/                  # unmodified starter-kit CSVs
│   └── splits/               # cached train/val/test arrays per dataset
├── src/
│   ├── data.py                 # [Rishi] load(), fixed split logic — single source of truth
│   ├── features/
│   │   ├── base.py             # [Min] ID + crossed features (FM-style)
│   │   ├── sequential.py       # [Nandit] per-user recent-interaction sequences
│   │   └── propensity.py       # [Vidush] exposure-propensity estimation for IPS
│   ├── models/
│   │   ├── fm.py                # baseline reproduction (numpy, starter-kit provided — do not edit)
│   │   ├── deepfm_mtl.py         # [Min] multi-task DeepFM — primary model
│   │   ├── seq_encoder.py        # [Nandit] GRU/attention over interaction history
│   │   ├── ssl_pretrain.py       # [Nandit] masked-item pretext task, opportunistic
│   │   └── ensemble.py           # [Sarthak] checkpoint blending, day-3 utility
│   ├── train.py                # single training run, writes checkpoint + val metrics
│   ├── evaluate.py             # GAUC / nDCG@5 / primary — starter-kit provided, DO NOT MODIFY
│   └── submit.py                # [Sarthak] writes/validates submission CSV
├── agent/
│   ├── orchestrator.py          # [Rishi] the outer iteration loop (see §6)
│   ├── hypothesis.py            # proposes next experiment given log history
│   └── convergence.py           # [Rishi] ε/N check + iteration/wall-clock cap
├── logs/
│   └── run_log.jsonl            # [Rishi owns schema] one line per iteration, append-only (see §7)
├── configs/
│   ├── kuairand_pure.yaml       # [Rishi owns seed/split values]
│   ├── kuairand_1k.yaml
│   └── kuairand_27k.yaml
├── scripts/
│   └── check.sh                 # required pre-PR check — everyone runs this before pushing
├── checkpoints/
└── outputs/
    ├── submission_pure.csv
    ├── submission_1k.csv        # bonus, optional
    └── submission_27k.csv       # bonus, optional
```

---

## 5. Interface contracts — where the five pieces meet

These are load-bearing. Breaking a contract without updating this file and
notifying the other affected role is the single most likely cause of a
failed integration.

| Interface | Producer | Consumer(s) | Contract |
|---|---|---|---|
| `data.load(config)` → arrays | Rishi | everyone | Fixed shape/dtype, documented in `src/data.py` docstring. Never re-implement or re-derive the split elsewhere. |
| `DeepFMMTL.forward(x, sample_weight=None, seq_embedding=None)` | Min | Vidush, Nandit | **Both optional keyword arguments must exist in Min's model from day 1**, even before Vidush/Nandit's code is ready. Vidush's IPS weights plug into `sample_weight`; Nandit's sequence features plug into `seq_embedding`. If Min changes this signature, Vidush and Nandit must be notified in the same message as the commit. |
| Checkpoint format | Min, Nandit | Sarthak | Every saved checkpoint is a dict: `{"state_dict": ..., "config": ..., "val_metrics": {...}}`. No per-model special-casing — if your model can't save in this shape, wrap it until it can. |
| `run_log.jsonl` schema | Rishi (writer via orchestrator) | Sarthak (reader for results table) | Exact schema in §7. Sarthak's results table and resource summary are generated *from this file*, never hand-typed. |

If you are about to change any of the above signatures or schemas, stop
and flag it — don't push a breaking change silently.

---

## 6. The outer agent loop (`agent/orchestrator.py`)

This only applies when operating as Rishi's role, or when any role is
running their own local iterations before merging. Each iteration does
exactly this, in order:

1. **Read state** — load `logs/run_log.jsonl`, get current best validation
   primary score and what's been tried.
2. **Propose a hypothesis** — one sentence: what will change and why,
   grounded in the priority list in §8. Do not repeat a hypothesis already
   tried and rejected without new justification.
3. **Implement the change** — as a code diff, not a full rewrite, unless
   this is the first iteration for a new model family.
4. **Train** — `src/train.py` with the current config. Backprop happens
   here, internally, per the model — this is separate from the outer
   loop's convergence check below.
5. **Evaluate** — `src/evaluate.py` on validation only. Never touch test.
6. **Log** — append one entry to `run_log.jsonl` (schema in §7), whether
   the iteration succeeded, failed, or was a wash.
7. **Check convergence** — has validation primary improved by > ε=0.002 in
   any of the last N=3 iterations? Has the 50-iteration or 6h wall-clock
   cap been hit? If yes to any → stop; the best checkpoint becomes the
   candidate for final submission.
8. **If not converged** → return to step 1.

**On failure inside any step** (bad hyperparameter, NaN loss, timeout):
catch it, log the error and the recovery attempted (retry with adjusted
setting / skip and revert to last-known-good config / reduce batch size),
and continue the loop — do not halt. This is what gets graded as
Robustness: not "never fails," but "fails gracefully and keeps going."

---

## 7. Run log schema (`logs/run_log.jsonl`) — one JSON object per line

```json
{
  "iteration": 7,
  "timestamp": "2026-08-29T14:03:00Z",
  "role": "Min",
  "dataset": "kuairand_pure",
  "hypothesis": "Add play_time as auxiliary head alongside click; expect primary +0.005 from shared embedding regularization",
  "code_diff_ref": "diffs/iter_007.patch",
  "model": "deepfm_mtl",
  "metrics": {"gauc": 0.6702, "ndcg5": 0.5401, "primary": 0.6052},
  "delta_vs_prev_best": 0.0031,
  "error": null,
  "recovery_action": null,
  "manual_intervention": false,
  "wall_clock_sec": 812,
  "tokens_used": {"input": 4200, "output": 1800}
}
```

- `role` must be one of the five names in §1 — this is what lets Sarthak's
  results table attribute progress correctly when logs from all five
  branches are merged.
- If `error` is non-null, `recovery_action` must describe what was done
  about it.
- If a human stepped in to fix/redirect/override the loop, set
  `manual_intervention: true` and describe the intervention — this feeds
  the autonomy summary directly and must be honest, not minimized.
- Timestamps are UTC (see §3).

---

## 8. Techniques — priority order (do these in this order unless the log
history says otherwise)

**Priority 1 — Multi-task learning (Min).** DeepFM with a primary head for
`long_view` and auxiliary heads for `click`, `like`, `play_time` (or a
subset). Shared embeddings should lift the primary task even though only
it is scored. Replaces the FM baseline as the main model. Highest expected
ROI, lowest implementation risk — do this first, and expose the
`sample_weight`/`seq_embedding` hooks (§5) immediately, before the model
is otherwise "finished."

**Priority 2 — Counterfactual / IPS debiasing (Vidush).** Use KuaiRand's
randomized-exposure subset to estimate exposure propensities and reweight
training examples with inverse propensity scoring. This corrects for the
fact that most logged interactions reflect what a prior recommender
already chose to show. Read Zhao et al. (KDD 2024, "Counteracting Duration
Bias in Video Recommendation via Counterfactual Watch Time") for the
framing — do not import their code directly (different torch version,
different label definition). This is the most KuaiRand-specific,
highest-originality technique — treat it as the centerpiece for the
Innovation criterion.

**Priority 3 — Sequential/session features (Nandit).** A lightweight GRU
or single-layer attention encoder over each user's recent interaction
history, producing an embedding that feeds into Min's model via the
`seq_embedding` hook.

**Priority 4 — SSL pretraining (Nandit, opportunistic, Day 3 only).**
Masked-item prediction over interaction sequences, used only to pretrain
embeddings before supervised fine-tuning. Labels aren't scarce here, so
expected lift is modest — run as one logged experiment, keep only if it
beats the multi-task model without it. First thing to cut if time is
short.

**Priority 5 — Ensembling (Sarthak, Day 3, low-risk wrap-up).** Blend
predictions from the top 2–3 checkpoints already produced by the loop
(weighted average or rank-average). Uses outputs the loop already
generated — good final iteration before freezing the submission.

---

## 9. Evaluation rules — use starter-kit `evaluate.py`, never reimplement

- GAUC: per-user AUC, weighted by positive count, only users with
  0 < positives < impressions.
- nDCG@5: gain = 2^rel − 1; users with zero positives score 0 and are
  included in the average.
- primary = mean(GAUC, nDCG@5).
- Ceiling context: perfect ranking → primary ≈ 0.8645 (not 1.0, because
  27.1% of users have no positive label). Random ≈ 0.4753. Official
  baseline = 0.5946 (hidden test) / 0.6016 (validation). Judge progress
  against 0.8645, not 1.0.

---

## 10. Submission

`src/submit.py --check` before finalizing, always. Required columns:
`row_id,user_id,video_id,score`. `row_id` is the join key (not
`user_id`+`video_id` — those repeat). Rejects on header mismatch,
row-count mismatch, row_id gaps, misalignment, or non-numeric/NaN/Inf
scores.

---

## 11. Git & CI workflow

- Branch off `main` only (never off another role's branch).
- Run `scripts/check.sh` locally before every push.
- Open a PR against `main`. The required GitHub Actions check
  (`.github/workflows/check.yml`) must pass before Rishi reviews.
- **Merge order:** Min's branch first (base model), then Vidush and Nandit
  against Min's already-merged code, then Sarthak's last (needs
  checkpoints from the others to exist).
- Only Rishi merges into `main`.

```bash
# scripts/check.sh — run this before every push
python -m pytest tests/ -x
python src/train.py --config configs/kuairand_pure.yaml --smoke-test
python src/evaluate.py --checkpoint checkpoints/smoke_test.pt --split val
python src/submit.py --check --file outputs/smoke_submission.csv
```

`--smoke-test` runs one iteration on a ~5,000-row subsample — fast enough
for any machine, sufficient to catch shape mismatches, path errors, and
import failures before they reach review.

---

## 12. Day-by-day plan and sync points

**Day 1 (Rishi builds, everyone else scaffolds):** Reproduce the official
FM baseline exactly (validate against published validation scores: GAUC
0.6674 / nDCG@5 0.5357 / primary 0.6016). Rishi builds
`data.py`/`orchestrator.py`/`convergence.py`/logging. Min/Vidush/Nandit
scaffold their own files against a stubbed interface until Rishi's real
one lands, then confirm `--smoke-test` passes against real data.
Sarthak sets up submission validation and bonus-dataset configs.
End of day: 15-minute sync — interface freeze confirmed by all five.

**Day 2 (parallel technique work + integration):** Min/Vidush/Nandit run
their own logged iterations against Priorities 1–3. Midday: async
check-in — each posts their current best primary score; Rishi decides
what to prioritize for the merge. End of day: Rishi merges Min's branch,
then attempts Vidush's and Nandit's on top, pairing immediately with
anyone whose branch doesn't merge cleanly.

**Day 3 (converge, opportunistic extras, package):** Feature freeze in the
morning — no new techniques except Nandit's SSL trial and Sarthak's
ensembling. Run to convergence on KuaiRand-Pure, generate and validate the
final submission. If time remains, attempt KuaiRand-1k with the same
pipeline (skip 27k — too large to safely debug in remaining time). Evening
sync: review the combined `run_log.jsonl` together for an honest
manual-intervention count; Sarthak generates the results table and
resource summary from the log file; Rishi signs off before packaging.

---

## 13. Definition of done

- [ ] Baseline reproduced and matches published numbers within noise
- [ ] Convergence criteria triggered (not just the iteration cap)
- [ ] `run_log.jsonl` has one clean entry per iteration, no gaps, correct
      `role` attribution
- [ ] Final submission passes `submit.py --check`
- [ ] Results table with delta vs. baseline (GAUC, nDCG@5, primary)
- [ ] Resource summary: total tokens, wall-clock, iteration count,
      GPU-hours if any
- [ ] Manual intervention count stated explicitly, with reasons
- [ ] Every role's branch passed `scripts/check.sh` before merge
- [ ] No absolute paths anywhere in the merged codebase
- [ ] `requirements.txt` / `.python-version` verified identical across all
      five machines
- [ ] Min's `sample_weight` / `seq_embedding` hooks are actually used by
      Vidush's and Nandit's merged code, not sitting unused
- [ ] Final submission passes `submit.py --check` on a machine other than
      the one it was generated on

---

## 14. If something goes wrong — do this, not that

- **CI fails on a PR:** fix locally and re-push. Do not merge with a
  failing check, and do not modify `.github/workflows/check.yml` to make
  a failing check pass — that hides the problem instead of fixing it.
- **A merge conflict appears in a shared file** (e.g. `configs/*.yaml`):
  stop and involve Rishi directly — don't resolve it unilaterally if the
  conflict touches seed values, split logic, or convergence thresholds.
- **Your model doesn't reproduce a teammate's reported score:** check
  `.python-version`, `requirements.txt`, and the config's seed value
  before assuming the model itself is wrong. This is almost always an
  environment or seed mismatch, not a modeling bug.
- **You're unsure whether a technique/library/pretrained weight is
  allowed:** don't use it — flag it to the human operator and wait for a
  decision, per §2.
- **You're asked to do something outside your role's owned files:** say so
  explicitly and ask for confirmation before proceeding, per §0.
