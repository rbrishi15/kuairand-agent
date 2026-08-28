"""[Vidush — PLACEHOLDER] Priority 2: exposure-propensity estimation for IPS.

Use KuaiRand's randomized-exposure log (log_random_4_22_to_5_08_pure.csv,
~1.18M rows) to estimate exposure propensities and reweight training
examples via inverse propensity scoring. Framing: Zhao et al. (KDD 2024,
"Counteracting Duration Bias in Video Recommendation via Counterfactual
Watch Time") — read for the idea, do not import their code (different
torch version, different label definition; CLAUDE.md §8).

Output plugs into DeepFMMTL.forward(x, sample_weight=...) — that hook
already exists in src/models/deepfm_mtl.py.
"""


def estimate_propensity(splits, random_log_path):
    raise NotImplementedError('TODO(Vidush): propensity model + IPS weight computation')
