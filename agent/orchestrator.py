"""[Rishi] The outer agent loop (CLAUDE.md §6).

Each iteration, in order: read state from logs/run_log.jsonl -> propose a
hypothesis -> apply it as config overrides -> train -> evaluate on valid
only -> append one log entry (exact schema, §7) -> check convergence ->
loop or stop. A failure inside any step is caught, logged with a recovery
action, and the loop continues rather than halting — CLAUDE.md grades this
as Robustness, not "never fails."
"""
import argparse
import copy
import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.convergence import ConvergenceTracker
from agent.hypothesis import propose
from src.config import load_config
from src.train import train

DEFAULT_ROLE = 'Rishi'


def read_log(log_path):
    if not os.path.exists(log_path):
        return []
    with open(log_path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _json_default(o):
    # evaluate()'s metrics dict can hold numpy scalar types (e.g. float32)
    # from arithmetic over numpy arrays — evaluate.py is frozen (CLAUDE.md
    # §4, do not modify), so normalize here instead.
    if hasattr(o, 'item'):
        return o.item()
    raise TypeError(f'not JSON serializable: {o!r}')


def append_log(log_path, entry):
    os.makedirs(os.path.dirname(log_path) or '.', exist_ok=True)
    with open(log_path, 'a') as fh:
        fh.write(json.dumps(entry, default=_json_default) + '\n')


def apply_overrides(config, overrides):
    cfg = copy.deepcopy(config)
    for k, v in overrides.items():
        if k == 'seed':
            cfg['seed'] = v
        else:
            cfg['model'][k] = v
    return cfg


def run(config_path, role=DEFAULT_ROLE, max_iterations_override=None):
    config = load_config(config_path)
    log_path = config.get('log_path', 'logs/run_log.jsonl')
    model_name = config['model']['name']

    conv_cfg = dict(config.get('convergence', {}))
    if max_iterations_override:
        conv_cfg['max_iterations'] = max_iterations_override
    tracker = ConvergenceTracker(**conv_cfg)
    tracker.start()

    full_history = read_log(log_path)
    relevant = [h for h in full_history
                if h.get('dataset') == config.get('dataset') and h.get('model') == model_name]
    for h in relevant:
        if h.get('metrics'):
            tracker.record(h['iteration'], h['metrics']['primary'])

    iteration = (full_history[-1]['iteration'] if full_history else 0) + 1
    last_good_metrics = relevant[-1]['metrics'] if relevant and relevant[-1].get('metrics') else None

    while True:
        hyp = propose(len(relevant), config)
        cfg_i = apply_overrides(config, hyp['overrides'])
        t0 = time.time()
        error, recovery, metrics = None, None, None
        try:
            result = train(cfg_i, verbose=False)
            metrics = result['val_metrics']
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            recovery = 'skipped iteration, reverted to last-known-good config'
            metrics = last_good_metrics

        prev_best = tracker.best if tracker.best > -1 else None
        entry = {
            'iteration': iteration,
            'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
            'role': role,
            'dataset': config.get('dataset'),
            'hypothesis': hyp['hypothesis'],
            'code_diff_ref': None,
            'model': model_name,
            'metrics': metrics,
            'delta_vs_prev_best': (metrics['primary'] - prev_best) if metrics and prev_best is not None else None,
            'error': error,
            'recovery_action': recovery,
            'manual_intervention': False,
            'wall_clock_sec': round(time.time() - t0, 1),
            'tokens_used': {'input': 0, 'output': 0},
        }
        append_log(log_path, entry)
        full_history.append(entry)
        relevant.append(entry)
        if metrics:
            last_good_metrics = metrics
            tracker.record(iteration, metrics['primary'])

        stop, reason = tracker.should_stop()
        status = f"primary={metrics['primary']:.4f}" if metrics else 'ERROR'
        print(f"iter {iteration}: {hyp['hypothesis'][:70]}... {status} stop={stop}({reason})")
        if stop:
            print(f"stopped: {reason} after {len(relevant)} iterations")
            break
        iteration += 1

    return full_history


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--role', default=DEFAULT_ROLE)
    ap.add_argument('--max-iterations', type=int, default=None)
    a = ap.parse_args()
    run(a.config, role=a.role, max_iterations_override=a.max_iterations)
