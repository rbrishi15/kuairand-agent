"""[Vidush] Priority 2: exposure-propensity estimation for IPS.

KuaiRand-Pure ships a randomized-exposure log
(log_random_4_22_to_5_08_pure.csv, ~1.18M rows, `is_rand=1`) alongside the
production-policy ("standard") log. Every row in the random log was shown
to its user uniformly at random from the candidate pool, independent of
any recommender's ranking. That makes it an unbiased reference for "how
often would video v be exposed if there were no selection policy at all" —
exactly what a propensity score needs, and something almost no other
public rec-sys dataset provides (KuaiRand's whole reason for including it).

The standard log, by contrast, reflects what a prior production
recommender already chose to show — high-propensity (popular / already
favored) videos are systematically over-represented relative to the
random log, and everything else under-represented. Training on the
standard log as-is inherits that selection bias.

Method (item-level IPS, Schnabel et al. 2016 framing; Zhao et al. KDD'24
motivates the counterfactual angle but its code isn't reused, per
CLAUDE.md §8):

  1. p_std(v)  = P(video v is exposed | standard/production policy)
  2. p_rand(v) = P(video v is exposed | uniform/random policy)   [unbiased]
  3. propensity(v) = p_std(v) / p_rand(v)   — how much the production
     policy over- or under-exposes v relative to uniform.
  4. IPS weight(v) = 1 / propensity(v), clipped and self-normalized to
     mean 1 so the *training loss scale* is unaffected, only the relative
     weighting of examples. Reweighting the standard log this way makes
     training approximate what it would look like under uniform exposure.

Deliberately item-level (not per user x video): the two candidate pools
are ~7.5k videos each with heavy overlap, so per-video counts are dense
enough to estimate reliably; a per-(user, video) model would be sparse
and overfit-prone as a first cut. A logistic-regression propensity model
over user/video/context features (closer to Zhao et al.'s framing) is a
natural follow-up iteration, logged separately per CLAUDE.md §6-7 if it
beats this baseline.

No test-split dates are ever read here, even though only exposure
*presence* (user_id, video_id) is used, never an outcome label: the random
log's date range overlaps the test window, so counting is capped at the
latest date appearing in `splits['train'] | splits['valid']` — derived
from whatever split boundaries `splits` was actually built with (never a
hardcoded copy of configs/*.yaml's dates, which Rishi owns and may change,
CLAUDE.md §0.4).

Output plugs into DeepFMMTL.forward(x, sample_weight=...) — that hook
already exists in src/models/deepfm_mtl.py. Only the 'train' split is
reweighted; valid/test get weight 1.0 (evaluation must stay on the true,
un-reweighted distribution).
"""
import csv
from collections import Counter

import numpy as np

_ALPHA = 5.0                 # default Laplace smoothing pseudo-count per video
_CLIP = (0.1, 10.0)          # default weight clipping range — IPS variance control


def _max_safe_date(splits):
    """Latest date safe to read from the random log without touching any
    date that falls inside the test split — derived from the actual train
    + valid rows passed in, not a hardcoded constant."""
    return max(row[0] for name in ('train', 'valid') for row in splits[name])


def _count_random_exposures(random_log_path, max_date):
    counts = Counter()
    with open(random_log_path) as fh:
        for r in csv.DictReader(fh):
            if r.get('is_rand') == '1' and int(r['date']) <= max_date:
                counts[r['video_id']] += 1
    return counts


def _video_propensities(std_counts, rand_counts, alpha):
    n_std = sum(std_counts.values())
    n_rand = sum(rand_counts.values())
    if n_rand == 0:
        raise ValueError('no usable random-log rows (check is_rand column and date range)')

    videos = set(std_counts) | set(rand_counts)
    smooth = alpha * len(videos)
    return {
        v: ((std_counts.get(v, 0) + alpha) / (n_std + smooth))
           / ((rand_counts.get(v, 0) + alpha) / (n_rand + smooth))
        for v in videos
    }


def estimate_propensity(splits, random_log_path, alpha=_ALPHA, clip=_CLIP):
    """Return {split_name: np.ndarray[float32]} of per-row IPS sample weights,
    aligned index-for-index with each split's row list (and so with
    `src.data.encode`'s output for the same splits). Only 'train' carries
    real reweighting; other splits are all-ones.

    `alpha` (Laplace smoothing pseudo-count) and `clip` (weight clipping
    bounds) are exposed so the orchestrator can sweep them as logged
    hypotheses (CLAUDE.md §6-7) without editing this file.
    """
    train = splits['train']
    std_counts = Counter(row[2] for row in train)          # row[2] == video_id
    rand_counts = _count_random_exposures(random_log_path, _max_safe_date(splits))
    propensity = _video_propensities(std_counts, rand_counts, alpha)

    train_weights = np.array(
        [1.0 / propensity[row[2]] for row in train], dtype=np.float32
    )
    np.clip(train_weights, *clip, out=train_weights)
    train_weights *= len(train_weights) / train_weights.sum()   # mean == 1

    weights = {'train': train_weights}
    for name, rows in splits.items():
        if name != 'train':
            weights[name] = np.ones(len(rows), dtype=np.float32)
    return weights


def propensity_report(splits, random_log_path, alpha=_ALPHA, clip=_CLIP):
    """Diagnostics for a hypothesis write-up / sanity check: how much
    random-log support the train videos actually have, and the shape of
    the resulting weight distribution. Not used by training itself.
    """
    train = splits['train']
    std_counts = Counter(row[2] for row in train)
    rand_counts = _count_random_exposures(random_log_path, _max_safe_date(splits))

    train_videos = set(std_counts)
    covered = train_videos & set(rand_counts)

    weights = estimate_propensity(splits, random_log_path, alpha=alpha, clip=clip)['train']
    percentiles = np.percentile(weights, [0, 25, 50, 75, 100])

    return {
        'n_train_videos': len(train_videos),
        'n_covered_by_random_log': len(covered),
        'coverage_frac': len(covered) / len(train_videos),
        'weight_min': float(percentiles[0]),
        'weight_p25': float(percentiles[1]),
        'weight_median': float(percentiles[2]),
        'weight_p75': float(percentiles[3]),
        'weight_max': float(percentiles[4]),
    }
