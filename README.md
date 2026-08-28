# kuairand-agent

Autonomous ML research agent for the KuaiRand-Pure recommendation benchmark
— NUS-SYNAPXE-IMDA AI Innovation Challenge 2026.

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

1. Pairwise/listwise loss instead of pointwise logloss (ranking-aligned
   objective) — Min/Priority 1's multi-task lift plus this is the most
   promising combination.
2. User interaction sequences — completely unused today (Nandit, Priority 3).
3. Multi-task auxiliary labels (`is_click`, `is_like`, `play_time_ms`, ...).
4. Counterfactual watch-time modeling (Vidush's IPS work is adjacent to this).
5. Model capacity (DeepFM/DCN/xDeepFM) — lower priority; capacity isn't the
   bottleneck per the ablation above.

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
