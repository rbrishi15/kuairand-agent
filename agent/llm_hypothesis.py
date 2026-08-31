"""[Rishi] LLM-driven hypothesis proposer — an alternative to
agent/hypothesis.py's deterministic grid, for agent/orchestrator.py's
--proposer llm flag.

The safety boundary, deliberately: the LLM CHOOSES a value for a fixed,
pre-validated set of config keys — it never writes or executes code, never
sees a file path, never touches src/data.py's split logic or
src/evaluate.py. This is the scaled-down slice of the "AIDE-style"
autonomous-agent idea that's actually safe to run unattended: real
reasoning over real run history, zero risk of it corrupting the codebase
or leaking test-split access, because there's no code-execution surface
for it to misuse. A full AIDE rebuild — the LLM drafting/improving/
debugging actual training code in a tree search — is a much bigger,
riskier project (real sandboxing of arbitrary generated code) not
attempted here; see the README for why that tradeoff wasn't taken.

Requires ANTHROPIC_API_KEY (reads it via the anthropic SDK's default
client construction) unless a `client` is injected for testing. Raises a
clear error rather than silently falling back — agent/orchestrator.py's
caller decides whether to catch it and use the deterministic proposer
instead.
"""
import json

ALLOWED_OVERRIDES = {
    'fm': {
        'seed': int,
        'k': int,
        'lr': float,
    },
    'deepfm_mtl': {
        'seed': int,
        'embed_dim': int,
        'lr': float,
        'use_ips': bool,
        'use_seq': bool,
        'loss': ('pointwise', 'bpr'),
        'seq_max_len': int,
        'seq_embed_dim': int,
        'seq_hidden_dim': int,
        'pairs_per_epoch': int,
    },
}

SYSTEM_PROMPT = """You are choosing the next hyperparameter configuration to \
try for a recommendation model, given prior experiment results. You may \
ONLY select values for the allowed keys listed below — you cannot invent \
new keys, and you never write or execute code; you are choosing a \
configuration, nothing else.

Respond with a single JSON object and nothing else:
{{"hypothesis": "<one sentence: what you're trying and why, grounded in the actual numbers below>", "overrides": {{...}}}}

Allowed keys and their types for model '{model_name}':
{allowed}

Rules:
- overrides must be a subset of the allowed keys above; unknown keys or
  wrong types will be rejected and this iteration wasted — don't guess.
- Do not repeat a combination already tried in the history below without
  new justification.
- Ground your reasoning in the actual metrics shown, not generic advice.
"""


def _format_allowed(model_name):
    allowed = ALLOWED_OVERRIDES.get(model_name, {})
    return '\n'.join(f'  - {k}: {v}' for k, v in allowed.items())


def _format_history(history):
    if not history:
        return 'No prior iterations for this model yet.'
    lines = ['Prior iterations (most recent last):']
    for h in history[-15:]:
        m = h.get('metrics') or {}
        lines.append(f"  iter {h['iteration']}: {h['hypothesis']} -> primary={m.get('primary')}")
    return '\n'.join(lines)


def _validate_overrides(overrides, model_name):
    allowed = ALLOWED_OVERRIDES.get(model_name, {})
    if not isinstance(overrides, dict):
        raise ValueError(f'overrides must be a dict, got {overrides!r}')
    clean = {}
    for k, v in overrides.items():
        if k not in allowed:
            raise ValueError(f"LLM proposed disallowed key '{k}' for model '{model_name}' "
                              f"(allowed: {sorted(allowed)})")
        spec = allowed[k]
        if isinstance(spec, tuple):
            if v not in spec:
                raise ValueError(f"'{k}'={v!r} not one of {spec}")
        elif spec is bool:
            if not isinstance(v, bool):
                raise ValueError(f"'{k}' must be bool, got {v!r}")
        elif spec is int:
            if not isinstance(v, int) or isinstance(v, bool):
                raise ValueError(f"'{k}' must be int, got {v!r}")
        elif spec is float:
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise ValueError(f"'{k}' must be a number, got {v!r}")
        clean[k] = v
    return clean


def propose_llm(history, config, client=None, model='claude-sonnet-4-5-20250929'):
    """Same shape of return as agent.hypothesis.propose() -- {hypothesis,
    overrides} -- plus tokens_used, since this is the one proposer that
    actually costs tokens. Reasons over the full run history rather than a
    fixed grid index.

    `history`: the relevant (dataset, model)-filtered log entries, same
    list agent/orchestrator.py already tracks as `relevant`.
    `client`: inject a fake for testing; defaults to a real
    anthropic.Anthropic() (reads ANTHROPIC_API_KEY from the environment).
    """
    if client is None:
        import anthropic
        client = anthropic.Anthropic()

    model_name = config['model']['name']
    system_prompt = SYSTEM_PROMPT.format(model_name=model_name, allowed=_format_allowed(model_name))
    history_text = _format_history(history)

    resp = client.messages.create(
        model=model,
        max_tokens=500,
        system=system_prompt,
        messages=[{'role': 'user', 'content': history_text}],
    )
    text = resp.content[0].text.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f'LLM response was not valid JSON: {text!r}') from e

    if 'hypothesis' not in parsed or 'overrides' not in parsed:
        raise ValueError(f'LLM response missing required keys: {parsed!r}')

    overrides = _validate_overrides(parsed['overrides'], model_name)
    tokens_used = {'input': getattr(resp.usage, 'input_tokens', 0),
                   'output': getattr(resp.usage, 'output_tokens', 0)}
    return {'hypothesis': parsed['hypothesis'], 'overrides': overrides, 'tokens_used': tokens_used}
