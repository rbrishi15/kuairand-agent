"""[Rishi] Tests for agent/llm_hypothesis.py using a fake Anthropic client
-- no ANTHROPIC_API_KEY needed. These cover the part that actually matters
for safety: a malformed or out-of-allowlist LLM response must be rejected,
never silently accepted or partially applied.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from agent.llm_hypothesis import propose_llm, _validate_overrides


class _FakeUsage:
    def __init__(self, input_tokens=100, output_tokens=50):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeContentBlock:
    def __init__(self, text):
        self.text = text


class _FakeResponse:
    def __init__(self, text, usage=None):
        self.content = [_FakeContentBlock(text)]
        self.usage = usage or _FakeUsage()


class _FakeMessages:
    def __init__(self, response_text):
        self.response_text = response_text
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeResponse(self.response_text)


class _FakeClient:
    def __init__(self, response_text):
        self.messages = _FakeMessages(response_text)


CONFIG = {'model': {'name': 'deepfm_mtl'}}


def test_valid_response_is_parsed_and_validated():
    text = json.dumps({'hypothesis': 'Try seq with a bigger hidden dim',
                        'overrides': {'seed': 2, 'use_seq': True, 'seq_hidden_dim': 64}})
    client = _FakeClient(text)
    result = propose_llm([], CONFIG, client=client)
    assert result['overrides'] == {'seed': 2, 'use_seq': True, 'seq_hidden_dim': 64}
    assert result['tokens_used'] == {'input': 100, 'output': 50}


def test_disallowed_key_is_rejected():
    text = json.dumps({'hypothesis': 'x', 'overrides': {'batch_size': 4096}})
    client = _FakeClient(text)
    with pytest.raises(ValueError, match='disallowed key'):
        propose_llm([], CONFIG, client=client)


def test_wrong_type_is_rejected():
    text = json.dumps({'hypothesis': 'x', 'overrides': {'use_ips': 'yes'}})
    client = _FakeClient(text)
    with pytest.raises(ValueError, match='must be bool'):
        propose_llm([], CONFIG, client=client)


def test_invalid_enum_value_is_rejected():
    text = json.dumps({'hypothesis': 'x', 'overrides': {'loss': 'listwise'}})
    client = _FakeClient(text)
    with pytest.raises(ValueError, match='not one of'):
        propose_llm([], CONFIG, client=client)


def test_malformed_json_is_rejected():
    client = _FakeClient('not json at all')
    with pytest.raises(ValueError, match='not valid JSON'):
        propose_llm([], CONFIG, client=client)


def test_missing_required_keys_is_rejected():
    client = _FakeClient(json.dumps({'overrides': {}}))
    with pytest.raises(ValueError, match='missing required keys'):
        propose_llm([], CONFIG, client=client)


def test_history_is_included_in_the_prompt():
    history = [{'iteration': 5, 'hypothesis': 'base run', 'metrics': {'primary': 0.6033}}]
    text = json.dumps({'hypothesis': 'x', 'overrides': {}})
    client = _FakeClient(text)
    propose_llm(history, CONFIG, client=client)
    sent = client.messages.last_kwargs['messages'][0]['content']
    assert 'iter 5' in sent and '0.6033' in sent


def test_unallowed_model_gives_no_valid_keys():
    with pytest.raises(ValueError, match='disallowed key'):
        _validate_overrides({'anything': 1}, 'unknown_model')


def test_valid_hidden_dims_becomes_a_tuple():
    overrides = _validate_overrides({'hidden_dims': [128, 64]}, 'deepfm_mtl')
    assert overrides == {'hidden_dims': (128, 64)}


def test_hidden_dims_wrong_shape_is_rejected():
    with pytest.raises(ValueError, match='hidden_dims'):
        _validate_overrides({'hidden_dims': [128, 64, 32, 16, 8]}, 'deepfm_mtl')  # too many layers


def test_hidden_dims_out_of_range_unit_is_rejected():
    with pytest.raises(ValueError, match='hidden_dims'):
        _validate_overrides({'hidden_dims': [4096]}, 'deepfm_mtl')  # exceeds max units


def test_hidden_dims_non_list_is_rejected():
    with pytest.raises(ValueError, match='hidden_dims'):
        _validate_overrides({'hidden_dims': 64}, 'deepfm_mtl')


def test_new_numeric_and_ips_alpha_overrides_are_accepted():
    overrides = _validate_overrides(
        {'epochs': 20, 'bs': 4096, 'patience': 3, 'dropout': 0.2,
         'weight_decay': 0.001, 'ips_alpha': 2.5},
        'deepfm_mtl',
    )
    assert overrides == {'epochs': 20, 'bs': 4096, 'patience': 3, 'dropout': 0.2,
                          'weight_decay': 0.001, 'ips_alpha': 2.5}
