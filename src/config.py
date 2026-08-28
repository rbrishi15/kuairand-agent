"""[Rishi] Load a configs/*.yaml file into a plain dict.

Every entrypoint (train.py, orchestrator.py, submit.py, eval_checkpoint.py)
reads its settings through this — never hardcode a path or hyperparameter
in a script body (CLAUDE.md §2/§3).
"""
import yaml


def load_config(path):
    with open(path) as fh:
        return yaml.safe_load(fh)
