"""[Sarthak] Generate and check submission files.

提交格式（CSV，含表头）：
    row_id,user_id,video_id,score

  row_id   : 0 起的行号，对应 data.load()[split] 的行序（确定性：先读
             log_standard_4_08_to_4_21_pure.csv 再读 log_standard_4_22_to_5_08_pure.csv，
             按 date 过滤后保持原文件顺序）
  user_id  : 该行的 user_id（冗余字段，仅用于校验对齐）
  video_id : 该行的 video_id（冗余字段，仅用于校验对齐）
  score    : 你的模型给该行打的分，任意实数，只用相对大小

为什么带 row_id：(user_id, video_id) 在评测集里**不唯一**
（test 集有 3.06% 的重复对，最多重复 12 次），所以无法作为主键。

用法：
    python3 src/submit.py --make   --config configs/kuairand_pure.yaml --split test outputs/submission.csv
    python3 src/submit.py --check  --config configs/kuairand_pure.yaml --split test outputs/submission.csv
    python3 src/submit.py --score  --config configs/kuairand_pure.yaml --split valid outputs/submission.csv

Ported from the starter kit's submit.py — `--data_dir` became `--config`
(data.load() now takes a config dict per CLAUDE.md §5's frozen interface,
not a raw path). `--make` defaults to retraining the official FM baseline
from scratch (same seed/hyperparameters as before, for a zero-checkpoint
sanity path — this is what scripts/check.sh's smoke test exercises).

Pass one or more `--checkpoint path` to score from a saved checkpoint
instead of retraining: a single `--checkpoint` scores that model directly
(via src/scoring.py's score(), the same helper scripts/eval_checkpoint.py
uses); more than one blends via src/models/ensemble.py's rank-average
ensemble_predict() (the same helper scripts/eval_ensemble.py uses), with
optional `--weight` per checkpoint (uniform if omitted). This is how the
actual best-checkpoint or ensemble submission gets made, e.g.:

    python3 src/submit.py --make --config configs/kuairand_pure_deepfm_mtl.yaml \
        --split test outputs/submission.csv \
        --checkpoint checkpoints/deepfm_mtl_seed0.pt \
        --checkpoint checkpoints/deepfm_mtl_seed1.pt \
        --checkpoint checkpoints/deepfm_mtl_seed2.pt \
        --checkpoint checkpoints/deepfm_mtl_seed3.pt \
        --checkpoint checkpoints/deepfm_mtl_seed4.pt
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Success messages below use non-ASCII (Chinese text, U+2713 checkmark);
# Windows consoles default to a codepage (e.g. cp1252) that can't encode
# them, crashing after all real validation has already succeeded. UTF-8
# stdout is safe cross-platform and required by CLAUDE.md's Every-machine
# reproducibility guarantee (Windows contributors couldn't otherwise ever
# see this script report success).
try:
    sys.stdout.reconfigure(encoding='utf-8')
except (AttributeError, ValueError):
    pass

from src.config import load_config
from src.data import load, encode
from src.evaluate import evaluate

HEADER = ['row_id', 'user_id', 'video_id', 'score']


def write_submission(path, rows, scores):
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        for i, (x, s) in enumerate(zip(rows, scores)):
            w.writerow([i, x[1], x[2], f"{float(s):.6g}"])


def read_submission(path, rows):
    """读取并逐行校验对齐，返回 scores。任何不一致都抛出可读错误。"""
    with open(path, newline='') as fh:
        r = csv.reader(fh)
        head = next(r, None)
        if head != HEADER:
            raise ValueError(f"表头必须是 {','.join(HEADER)}，实际是 {head}")
        scores, n = [], 0
        for ln, rec in enumerate(r, start=2):
            if len(rec) != 4:
                raise ValueError(f"第 {ln} 行有 {len(rec)} 个字段，应为 4 个")
            rid, uid, vid, sc = rec
            if int(rid) != n:
                raise ValueError(f"第 {ln} 行 row_id={rid}，应为 {n}（必须 0 起连续递增）")
            if n >= len(rows):
                raise ValueError(f"提交行数超过评测集（评测集 {len(rows)} 行）")
            if uid != rows[n][1] or vid != rows[n][2]:
                raise ValueError(f"第 {ln} 行对齐错误：提交 ({uid},{vid})，"
                                 f"评测集第 {n} 行是 ({rows[n][1]},{rows[n][2]})")
            try:
                v = float(sc)
            except ValueError:
                raise ValueError(f"第 {ln} 行 score 无法解析为数字：{sc!r}")
            if v != v or v in (float('inf'), float('-inf')):
                raise ValueError(f"第 {ln} 行 score 是 NaN/Inf，不允许")
            scores.append(v); n += 1
    if n != len(rows):
        raise ValueError(f"提交 {n} 行，评测集 {len(rows)} 行，数量不符")
    return scores


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('path')
    ap.add_argument('--config', default='configs/kuairand_pure.yaml')
    ap.add_argument('--split', default='test', choices=['valid', 'test'])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--make',  action='store_true', help='生成提交（默认用官方 FM baseline；见 --checkpoint）')
    g.add_argument('--check', action='store_true', help='只校验格式与对齐')
    g.add_argument('--score', action='store_true', help='校验并打分')
    ap.add_argument('--checkpoint', action='append', dest='checkpoints',
                     help='仅配合 --make：从已保存的 checkpoint 打分，而不是重新训练 FM；'
                          '重复该参数即可做 rank-average 集成（见 src/models/ensemble.py）')
    ap.add_argument('--weight', action='append', type=float, dest='weights',
                     help='每个 --checkpoint 对应一个权重，顺序一致；不传则均匀加权')
    a = ap.parse_args()

    config = load_config(a.config)
    splits = load(config)
    rows = splits[a.split]

    if a.make:
        enc, dim = encode(splits)
        X, y, u = enc[a.split]
        if a.checkpoints:
            if a.weights and len(a.weights) != len(a.checkpoints):
                raise SystemExit(f'{len(a.weights)} --weight flags but '
                                  f'{len(a.checkpoints)} --checkpoint flags')
            from src.models.ensemble import ensemble_predict
            scores = ensemble_predict(a.checkpoints, X, dim, u, splits=splits,
                                       split_name=a.split, weights=a.weights)
            source = f"{len(a.checkpoints)} checkpoint(s), rank-averaged"
        else:
            from src.models.fm import run_fm
            model, _ = run_fm(enc, dim, k=16, lr=0.001, seed=0)
            scores = model.predict(X)
            source = "official FM baseline (retrained)"
        write_submission(a.path, rows, scores)
        print(f"已写出 {a.path}：{len(rows):,d} 行（split={a.split}，source={source}）")
    else:
        scores = read_submission(a.path, rows)
        print(f"✓ 格式与对齐校验通过：{len(scores):,d} 行，split={a.split}")
        if a.score:
            r = evaluate([x[1] for x in rows], [x[6] for x in rows], scores)
            print(f"  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
