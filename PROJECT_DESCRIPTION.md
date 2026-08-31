# Project description (Devpost)

Source content for the Devpost written submission. Paste into the
Devpost form directly; this file exists so the content is drafted from
the actual repo state rather than written from memory.

## How our solution addresses the problem statement

The challenge asks for an autonomous ML research agent that improves a
recommendation model's score on KuaiRand through its own iteration loop,
not just a hand-tuned model. We built that as a real, running system,
not a simulation of one:

- **The outer loop is real code, not narrative.** `agent/orchestrator.py`
  implements the full cycle CLAUDE.md specifies: read prior results ->
  propose a hypothesis -> apply it as a config change -> train -> evaluate
  on validation only -> log one structured entry -> check convergence ->
  repeat. It ran to genuine convergence on KuaiRand-Pure (25 iterations,
  stopped because validation primary stopped improving by more than
  epsilon=0.002 over 3 consecutive iterations — not because it hit the
  50-iteration cap).
- **Two swappable hypothesis proposers.** A deterministic per-model grid
  (`agent/hypothesis.py`, no API key needed, always available) and an
  LLM-driven proposer (`agent/llm_hypothesis.py`) that calls the Claude
  API to reason over real run history and choose the next experiment from
  a fixed, type-checked allowlist of config keys — deliberately scoped so
  the LLM can never write or execute code, never touches the frozen
  split/evaluation logic, and can never leak test-split access.
- **Failure is handled, not avoided.** When the LLM proposer hit a real
  missing-credentials error live, the orchestrator logged the exact
  exception, fell back to the grid proposer, and completed the iteration
  normally — a genuine demonstration of "fails gracefully and keeps
  going," which is what the grading rubric's Robustness criterion asks
  for, not "never fails."
- **A genuinely KuaiRand-specific technique, not a generic recommender
  trick.** KuaiRand-Pure uniquely ships a randomized-exposure log
  alongside its production-policy log. We used that to estimate real
  per-video exposure propensities and built inverse-propensity-scored
  (IPS) sample weights, plus a genuinely unbiased off-policy evaluation
  set built from that same random log — so IPS's actual effect could be
  checked against ground truth instead of asserted.
- **Honest reporting over headline numbers.** Every technique we tried
  (multi-task DeepFM, IPS reweighting, a GRU sequence encoder, pairwise
  BPR loss, rank-average ensembling) is reported with multi-seed std, not
  single-run cherry-picking — including techniques that turned out flat
  or slightly negative. The README documents what didn't work as
  carefully as what did.

## Development tools used

- **Claude Code** (Anthropic's agentic CLI) — used throughout for
  implementation, debugging, and running experiments across multiple
  team members' branches.
- **VS Code** — primary editor.
- **Git / GitHub** — version control, branch-per-role workflow, PR-gated
  merges into `main` with a required CI check
  (`.github/workflows/check.yml`).
- **Docker** — `Dockerfile` pins the exact runtime environment
  (`python:3.12-slim` + `requirements.txt`) as the cross-platform
  tiebreaker when a result doesn't reproduce on a teammate's machine.

## APIs used

- **Anthropic Claude API** (`claude-sonnet-4-5-20250929`, via the
  `anthropic` Python SDK) — powers the optional LLM-driven hypothesis
  proposer (`agent/llm_hypothesis.py`, `--proposer llm`). Not used for
  any other part of the pipeline (data loading, training, and evaluation
  are all plain PyTorch/NumPy, no LLM calls).

## Libraries and frameworks used

- **PyTorch** (`torch==2.9.1`) — `DeepFMMTL` (shared-embedding FM +
  deep tower + multi-task heads), the GRU sequence encoder, and all
  training loops (pointwise BCE and pairwise BPR). CPU-only by design
  (CLAUDE.md's cross-platform requirement) — GPU is an optional speedup,
  never assumed.
- **NumPy** (`numpy==2.1.3`) — data loading/encoding (`src/data.py`),
  the item2vec-style SSL pretraining pretext task
  (`src/models/ssl_pretrain.py`), and all evaluation math.
- **PyYAML** (`pyyaml==6.0.2`) — config file loading.
- **pytest** (`pytest==8.3.4`) — the test suite (11 files, ~60 tests)
  gating every push via `scripts/check.sh`.
- **anthropic** (`anthropic==1.2.0`) — official Python SDK for the
  Claude API, used only by the optional LLM proposer.

## Datasets and assets used

- **KuaiRand-Pure** (required benchmark) — the official randomized-
  exposure + production-policy logs, video/user feature tables, from
  Zenodo (record 10439422). Used exactly as distributed, split by the
  fixed date ranges the task specifies (train 4/8-4/21, valid 4/22-4/28,
  test 4/29-5/8); no external or augmenting data.
- **KuaiRand-1K** (bonus benchmark, attempted) — same pipeline, same
  splits, run against the larger/denser 1K version of the dataset.
- **KuaiRand-27K** (bonus benchmark) — download/attempt status: see
  README's bonus-benchmark section for the honest current state.
- No manually labelled data, no synthetic data, and no data outside the
  official KuaiRand releases — CLAUDE.md's hard constraint (§2) is "no
  external training data," and nothing in this project uses any.
