#!/usr/bin/env bash
# Required pre-PR check (CLAUDE.md §11, playbook §4.3). Run this before
# every push; a PR that fails it doesn't get reviewed, let alone merged.
#
# Two fixes vs. the illustrative version in CLAUDE.md/the playbook (see
# the plan / commit message for why): evaluate.py has no CLI (it's frozen,
# do-not-modify) so the eval step calls scripts/eval_checkpoint.py instead;
# and submit.py's real CLI takes a positional path with --check/--make
# flags, not --file, and needs a --make step first since nothing else
# creates outputs/smoke_submission.csv.
#
# Also smoke-tests configs/kuairand_pure_deepfm_mtl_full.yaml (Min's model
# with Vidush's IPS weights AND Nandit's sequence encoder both enabled) —
# without this, CI only ever exercised the FM path and a plain merge could
# silently break the actual integration between the three techniques.
set -euo pipefail

python -m pytest tests/ -x
python src/train.py --config configs/kuairand_pure.yaml --smoke-test
python scripts/eval_checkpoint.py --checkpoint checkpoints/smoke_test.pt --split valid
python src/train.py --config configs/kuairand_pure_deepfm_mtl_full.yaml --smoke-test
python scripts/eval_checkpoint.py --checkpoint checkpoints/smoke_test.pt --split valid
python src/submit.py --make  --config configs/kuairand_pure.yaml --split valid outputs/smoke_submission.csv
python src/submit.py --check --config configs/kuairand_pure.yaml --split valid outputs/smoke_submission.csv
