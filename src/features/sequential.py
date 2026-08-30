"""[Nandit] Priority 3: per-user recent-interaction sequences.

Turns the raw rows from ``src.data.load(config)`` into, per user, the ordered
list of ``video_id``s that user interacted with inside the **train window**.
That history is consumed two ways:

  * ``src.models.ssl_pretrain`` — item2vec (skip-gram) over the sequences to
    pretrain a video-embedding table (Priority 4).
  * ``src.models.seq_encoder.SeqEncoder`` — a GRU over a user's sequence,
    producing the per-user vector that plugs into
    ``DeepFMMTL.forward(x, seq_embedding=...)`` (Min's hook, CLAUDE.md §5).

Design notes
------------
* **Train-window only.** Sequences are built from ``splits['train']`` and
  nothing else, so a sequence used to score a valid/test row is strictly in
  that row's past — no leakage across the official split boundary. Cold users
  (present in valid/test but not train) get an all-PAD sequence.
* **Local video vocab.** ``src.data.encode()`` owns the global id space for
  the DeepFM embedding table and does not expose per-field vocabs. The
  sequence encoder does not need to share ids with that table — its output is
  a dense vector, not an id — so this module derives its own compact
  ``video_id -> 0..V-1`` vocab from the train rows. This is reading rows to
  build a vocab, not re-deriving the split (CLAUDE.md §5).
* **Ordering.** ``src.data.load()`` does not surface ``hourmin``, so rows are
  ordered by ``date`` with a stable sort that preserves each user's original
  row order within a date (the CSV order, which the starter-kit README
  documents as deterministic). This is a slightly coarser order than the
  prototype's ``(date, hourmin)`` sort; it only matters for co-occurrence
  windows inside a single day.

Row tuple layout (from ``src.data.load``): ``(date, user_id, video_id,
author_id, tab, duration_ms, label)``.
"""
import numpy as np

# Column indices into a raw row tuple from src.data.load().
_DATE, _USER, _VIDEO = 0, 1, 2


def build_video_vocab(splits):
    """``video_id -> contiguous int id`` over videos seen in the train split.

    Ids are ``0 .. V-1`` in first-seen order (train rows iterated in load
    order). ``V`` (i.e. ``len(vocab)``) is reserved as the PAD id by
    ``sequences_for_rows`` and ``SeqEncoder``.
    """
    vocab = {}
    for row in splits['train']:
        vid = row[_VIDEO]
        if vid not in vocab:
            vocab[vid] = len(vocab)
    return vocab


def build_sequences(splits, vocab=None, max_len=50):
    """Per-user ordered interaction history from the train window.

    Returns ``dict[user_id -> list[int]]`` where each list holds up to
    ``max_len`` **most recent** video-vocab ids (older items dropped first).
    Videos not in ``vocab`` are skipped; by construction every train video is
    in a vocab built by :func:`build_video_vocab`, so this only drops items
    when a caller passes a restricted vocab.

    ``vocab`` defaults to :func:`build_video_vocab` on ``splits``.
    """
    if vocab is None:
        vocab = build_video_vocab(splits)

    # Stable sort by date keeps each user's within-day row order (load order).
    train_rows = sorted(splits['train'], key=lambda r: r[_DATE])

    seqs = {}
    for row in train_rows:
        vid = vocab.get(row[_VIDEO])
        if vid is None:
            continue
        seqs.setdefault(row[_USER], []).append(vid)

    if max_len is not None:
        for user, seq in seqs.items():
            if len(seq) > max_len:
                seqs[user] = seq[-max_len:]
    return seqs


def sequences_for_rows(rows, user_seqs, pad_id, max_len=50):
    """Align per-user sequences to an arbitrary split's rows.

    ``rows``      : the raw-row list for a split (``splits['valid']`` etc.).
    ``user_seqs`` : output of :func:`build_sequences` (train-window histories).
    ``pad_id``    : id to left-pad short/absent sequences with — pass
                    ``len(vocab)`` (the reserved slot).

    Returns an ``int64`` array of shape ``(len(rows), max_len)``, left-padded,
    truncated to the last ``max_len`` items. A user with no train history
    (cold start) yields an all-PAD row, which ``SeqEncoder`` maps to a
    near-zero vector.
    """
    out = np.full((len(rows), max_len), pad_id, dtype=np.int64)
    for i, row in enumerate(rows):
        seq = user_seqs.get(row[_USER])
        if not seq:
            continue
        seq = seq[-max_len:]
        out[i, max_len - len(seq):] = seq
    return out
