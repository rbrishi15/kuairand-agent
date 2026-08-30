# Priority 3 / 4 — sequential features + SSL: port notes

Owner: Nandit (`feat/sequential-ssl`). This records what was ported from the
pre-merge prototype, what was intentionally left out, and the prototype's
own findings so they aren't lost.

## What landed in the pipeline

| File | Contents |
|---|---|
| `src/features/sequential.py` | `build_video_vocab`, `build_sequences` (per-user train-window video history), `sequences_for_rows` (left-padded per-row id arrays). Reads `src.data.load()` rows; derives its own compact video vocab (does **not** re-derive the split). |
| `src/models/ssl_pretrain.py` | `make_pairs` + `pretrain_item2vec` — NumPy skip-gram + negative sampling over the sequences, Adam. Returns a `(num_items+1, k)` table, PAD row left at init. CPU-only. Ported near-verbatim from the prototype's `ssl_pretrain.py`. |
| `src/models/seq_encoder.py` | `SeqEncoder(nn.Module)` — single-layer GRU (or `mean` pooling) over a user's item-embedding sequence → per-row vector for `DeepFMMTL.forward(seq_embedding=...)`. `out_dim` sizes Min's `seq_dim`. Optional item2vec warm-start via `pretrained_emb`. Handles all-PAD (cold-start) rows → near-zero vector. |
| `experiments/run_seq.py` | Scratch runner (CLAUDE.md §6, local iterations pre-merge). Trains `DeepFMMTL` + `SeqEncoder` jointly on the frozen `src.data` / `src.evaluate` path; appends one `logs/run_log.jsonl` entry per run (`role: "Nandit"`). `--no-seq` gives the same-loop ablation baseline; `--ssl-warmstart` adds Priority 4. Does not modify `src/train.py` or `src/models/deepfm_mtl.py`. |

## Status: code-only, not yet executed

Local environment does not satisfy CLAUDE.md §3: `.python-version` pins
**3.12**, only 3.9 / 3.14 are installed, and `torch==2.9.1` has no 3.14
wheels. NumPy-only logic (`sequential.py`, `ssl_pretrain.py`) is
smoke-tested and passes; `seq_encoder.py` and `run_seq.py` are syntax-checked
only. **No `run_log.jsonl` entries have been written** — per §2 none of this
"counts" until it runs on a 3.12 + torch env:

```
python experiments/run_seq.py --no-seq                       # baseline row
python experiments/run_seq.py                                # + GRU seq encoder
python experiments/run_seq.py --ssl-warmstart --ssl-epochs 5 # + item2vec warm-start
```

## Deliberately NOT ported

The prototype was a standalone copy of the starter kit with its own
`data.py` / `evaluate.py` / `baseline.py`. Those reimplementations were
dropped — the shared `src/evaluate.py` (frozen) and `src/data.py` (Rishi's)
are canonical, and any delta must be measured against them.

The analysis scripts (`ssl_compare.py`, `ssl_stratified.py`,
`ssl_freeze_compare.py`, `ssl_overwrite_check.py`, `loss_compare.py`,
`ablation_features.py`) were **not** ported: they depend on starter-kit-only
APIs (`data.build_vocabs` / `data.field_layout`, `baseline.run_fm(init_V=,
frozen_rows=, loss=, return_model=)`) that don't exist in this repo and
aren't Nandit's files to add. Their findings are recorded below; re-run
against `src.*` in this directory later if a claim needs to be reproduced
on the canonical path.

## Prototype findings (from the standalone scripts — measured on their own
eval copy, treat as directional until reproduced here)

- **SSL warm-start of the FM `V` table ≈ noise on aggregate primary.**
  Averaged over seeds, random-init vs item2vec-warm-start FM differed by
  less than the seed std (~0.0008).
- **Small, consistent benefit only on rare videos.** Stratifying test rows
  by video train-frequency, the rarest tercile showed a small positive SSL
  delta with consistent sign across seeds; medium/popular did not.
- **Fine-tuning overwrites SSL structure ∝ update count.** cos(init, final)
  of video embeddings drops as train frequency rises; L2 movement from the
  SSL start correlates positively with frequency. Freezing the rare rows at
  their SSL init did not enlarge the rare-stratum gain.
- **Loss function is where the headroom was.** Swapping FM's pointwise
  logloss for pairwise BPR at a reduced lr (~2–3×10⁻⁴) gave test primary
  ≈ 0.596 vs ≈ 0.592 same-seed pointwise (~+0.004) at roughly half the
  variance. Listwise within-user softmax tied pointwise. **This is Min's
  lane (loss lives in the training loop), not Nandit's — flagged to
  Min/Rishi, not ported here.**
- **Extra static feature domains: no lift** (reproduces the note already in
  `src/features/base.py`).

## Implications for the merge

- The item2vec warm-start is expected to be marginal on its own; keep it in
  the run set as one logged experiment (Priority 4 is explicitly
  "opportunistic, keep only if it beats the plain MTL model").
- The GRU `seq_embedding` path (Priority 3) is the untested part and the
  reason this branch exists — it adds behavioural history the static fields
  don't carry, which none of the prototype's FM-warm-start experiments
  actually exercised.
- Rishi: `experiments/run_seq.py` needs folding into `src/train.py`'s
  `MODEL_REGISTRY` (e.g. `deepfm_mtl_seq`) at integration, with the seq
  arrays built inside `train()` alongside `encode()`.
