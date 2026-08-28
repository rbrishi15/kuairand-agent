"""[Sarthak — starter-kit ported, please take ownership] Generate and check
submission files.

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

Ported from the starter kit's submit.py — only two things changed from the
original, both mechanical: `--data_dir` became `--config` (data.load() now
takes a config dict per CLAUDE.md §5's frozen interface, not a raw path),
and `--make` now calls run_fm() directly instead of duplicating its training
loop inline (run_fm changed to return the trained model — see
src/models/fm.py's docstring). Same seed, same hyperparameters, same result.

TODO(Sarthak): `--make` retrains an FM from scratch every time. Once
checkpoints exist, prefer loading one via scripts/eval_checkpoint.py's
score() helper instead — faster, and lets you point --make at whichever
checkpoint (FM, DeepFM-MTL, ensemble) you actually want to submit.
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    g.add_argument('--make',  action='store_true', help='用官方 FM baseline 生成示例提交')
    g.add_argument('--check', action='store_true', help='只校验格式与对齐')
    g.add_argument('--score', action='store_true', help='校验并打分')
    a = ap.parse_args()

    config = load_config(a.config)
    splits = load(config)
    rows = splits[a.split]

    if a.make:
        from src.models.fm import run_fm
        enc, dim = encode(splits)
        X, y, u = enc[a.split]
        model, _ = run_fm(enc, dim, k=16, lr=0.001, seed=0)
        write_submission(a.path, rows, model.predict(X))
        print(f"已写出 {a.path}：{len(rows):,d} 行（split={a.split}，官方 FM baseline）")
    else:
        scores = read_submission(a.path, rows)
        print(f"✓ 格式与对齐校验通过：{len(scores):,d} 行，split={a.split}")
        if a.score:
            r = evaluate([x[1] for x in rows], [x[6] for x in rows], scores)
            print(f"  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
