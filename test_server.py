"""Tests for aisuite_mcp_server.

Running them:

    pip install pytest
    pytest -v

The tests never touch a real provider, so they need no network access or
keys: the keyring is faked and the aisuite client replaced with a double
that records how it was called.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path
from unittest import mock

import pytest

import aisuite_mcp_server as srv

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


class FakeResponse:
    """Mimics aisuite's response object."""

    def __init__(self, content: str):
        message = type('Message', (), {'content': content})()
        choice = type('Choice', (), {'message': message})()
        self.choices = [choice]


class FakeClient:
    """Client double that remembers how it was called."""

    def __init__(self, answer: str = 'hello', error: Exception | None = None):
        self.configs: list[dict] = []
        self.calls: list[dict] = []
        self._answer = answer
        self._error = error
        self_ = self

        class Completions:
            def create(self, **kwargs):
                self_.calls.append(kwargs)
                if self_._error:
                    raise self_._error
                return FakeResponse(self_._answer)

        class Chat:
            completions = Completions()

        self.chat = Chat()

    def configure(self, configs: dict) -> None:
        self.configs.append(configs)


def fake_keyring(values: dict[tuple[str, str], str | None]):
    """Replace keyring.get_password with a lookup in *values*."""
    return mock.patch.object(
        srv.keyring, 'get_password', side_effect=lambda s, u: values.get((s, u))
    )


# --------------------------------------------------------------------------
# provider_config
# --------------------------------------------------------------------------


def test_local_provider_needs_no_key():
    config, error = srv.provider_config('ollama')
    assert error is None
    assert config == {'timeout': 300}


def test_local_config_is_a_copy():
    """Modifying the result must not touch the module-level table."""
    config, _ = srv.provider_config('ollama')
    config['timeout'] = 1
    again, _ = srv.provider_config('ollama')
    assert again['timeout'] == 300


def test_single_key_is_renamed_to_aisuites_name():
    with fake_keyring({('mistral', 'api-key'): 'secret'}):
        config, error = srv.provider_config('mistral')
    assert error is None
    assert config == {'api_key': 'secret'}


def test_huggingface_uses_token_instead_of_api_key():
    """The keyring name is the same everywhere; aisuite wants "token" here."""
    with fake_keyring({('huggingface', 'api-key'): 'hf_abc'}):
        config, error = srv.provider_config('huggingface')
    assert error is None
    assert config == {'token': 'hf_abc'}


def test_missing_key_gives_a_message_with_the_command():
    with fake_keyring({}):
        config, error = srv.provider_config('mistral')
    assert config is None
    assert 'keyring set mistral api-key' in error


def test_multiple_missing_only_names_the_missing_fields():
    """Fields that are already set shouldn't appear in the fix instructions."""
    with fake_keyring({('watsonx', 'api-key'): 'secret'}):
        config, error = srv.provider_config('watsonx')
    assert config is None
    assert 'keyring set watsonx service-url' in error
    assert 'keyring set watsonx project-id' in error
    assert 'keyring set watsonx api-key' not in error


def test_multiple_fields_complete_yields_all_config_keys():
    with fake_keyring(
        {
            ('watsonx', 'service-url'): 'https://eu-de.ml.cloud.ibm.com',
            ('watsonx', 'api-key'): 'secret',
            ('watsonx', 'project-id'): 'p-123',
        }
    ):
        config, error = srv.provider_config('watsonx')
    assert error is None
    assert config == {
        'service_url': 'https://eu-de.ml.cloud.ibm.com',
        'api_key': 'secret',
        'project_id': 'p-123',
    }


def test_optional_field_may_be_missing():
    """azure works without api-version; that's listed in _OPTIONAL."""
    with fake_keyring(
        {('azure', 'endpoint'): 'https://x.azure.com', ('azure', 'api-key'): 'secret'}
    ):
        config, error = srv.provider_config('azure')
    assert error is None
    assert config == {'base_url': 'https://x.azure.com', 'api_key': 'secret'}


def test_unreachable_keyring_gives_a_message_not_an_exception():
    with mock.patch.object(
        srv.keyring, 'get_password', side_effect=RuntimeError('no backend')
    ):
        config, error = srv.provider_config('mistral')
    assert config is None
    assert 'keyring unreachable' in error


def test_unreachable_keyring_error_is_flagged_structurally():
    """check_setup.py branches on this flag instead of parsing the message."""
    with mock.patch.object(
        srv.keyring, 'get_password', side_effect=RuntimeError('no backend')
    ):
        _, error = srv.provider_config('mistral')
    assert error.keyring_unreachable is True


def test_missing_key_error_has_a_short_summary():
    """The summary is the one-liner check_setup.py's overview can show,
    without the multi-line 'keyring set ...' fix command."""
    with fake_keyring({}):
        _, error = srv.provider_config('mistral')
    assert error.keyring_unreachable is False
    assert error.summary == 'mistral is missing api-key in the keyring'
    assert 'keyring set' not in error.summary
    assert 'keyring set mistral api-key' in error


# --------------------------------------------------------------------------
# run_query
# --------------------------------------------------------------------------


def test_successful_call_returns_the_answer():
    client = FakeClient(answer='Hello there.')
    with fake_keyring({('mistral', 'api-key'): 'secret'}):
        output = json.loads(srv.run_query(client, 'mistral', 'small', 'hi'))
    assert output == {'answer': 'Hello there.'}
    assert client.calls[0]['model'] == 'mistral:small'
    assert client.calls[0]['messages'] == [{'role': 'user', 'content': 'hi'}]


def test_key_goes_through_configure_not_the_environment(monkeypatch):
    """The core of the security choice: nothing in os.environ."""
    monkeypatch.delenv('MISTRAL_API_KEY', raising=False)
    client = FakeClient()
    with fake_keyring({('mistral', 'api-key'): 'secret'}):
        srv.run_query(client, 'mistral', 'small', 'hi')

    assert client.configs == [{'mistral': {'api_key': 'secret'}}]
    import os

    assert 'MISTRAL_API_KEY' not in os.environ


def test_system_prompt_comes_first():
    client = FakeClient()
    with fake_keyring({('mistral', 'api-key'): 'secret'}):
        srv.run_query(client, 'mistral', 'small', 'hi', system_prompt='be brief')
    assert client.calls[0]['messages'][0] == {
        'role': 'system',
        'content': 'be brief',
    }


def test_temperature_is_only_sent_when_set():
    client = FakeClient()
    with fake_keyring({('mistral', 'api-key'): 'secret'}):
        srv.run_query(client, 'mistral', 'small', 'hi')
        assert 'temperature' not in client.calls[0]

        srv.run_query(client, 'mistral', 'small', 'hi', temperature=0.5)
        assert client.calls[1]['temperature'] == 0.5


def test_unknown_provider_is_caught_before_the_keyring():
    client = FakeClient()
    output = json.loads(srv.run_query(client, 'nonsense', 'x', 'hi'))
    assert 'unknown provider' in output['error']
    assert client.calls == []


def test_provider_error_becomes_a_message_not_a_crash():
    client = FakeClient(error=RuntimeError('connection refused'))
    with fake_keyring({('mistral', 'api-key'): 'secret'}):
        output = json.loads(srv.run_query(client, 'mistral', 'small', 'hi'))
    assert output['error'] == 'connection refused'


def test_long_error_message_gets_truncated():
    client = FakeClient(error=RuntimeError('x' * 2000))
    with fake_keyring({('mistral', 'api-key'): 'secret'}):
        output = json.loads(srv.run_query(client, 'mistral', 'small', 'hi'))
    assert len(output['error']) < 600
    assert output['error'].endswith('[...truncated]')


def test_missing_key_never_reaches_the_provider():
    client = FakeClient()
    with fake_keyring({}):
        output = json.loads(srv.run_query(client, 'mistral', 'small', 'hi'))
    assert 'keyring set' in output['error']
    assert client.calls == []


def test_the_question_does_not_end_up_in_the_log(caplog):
    """What the user asks shouldn't end up in stderr."""
    client = FakeClient(error=RuntimeError('broken'))
    with fake_keyring({('mistral', 'api-key'): 'secret'}):
        srv.run_query(client, 'mistral', 'small', 'is company X bankrupt?')
    assert 'bankrupt' not in caplog.text


def test_the_key_does_not_end_up_in_the_log(caplog):
    client = FakeClient(error=RuntimeError('broken'))
    with fake_keyring({('mistral', 'api-key'): 'verysecret123'}):
        srv.run_query(client, 'mistral', 'small', 'hi')
    assert 'verysecret123' not in caplog.text


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------


def test_provider_list_comes_from_aisuite_not_from_us():
    """This server shouldn't decide which providers exist."""
    from aisuite.provider import ProviderFactory

    assert set(srv.PROVIDERS) == set(ProviderFactory.get_supported_providers())


def test_list_contains_providers_in_no_table_at_all():
    """Proves the list isn't secretly derived from the tables."""
    tabled = srv._KEY_FIELDS.keys() | srv._LOCAL.keys() | srv._OWN_CREDENTIALS
    assert set(srv.PROVIDERS) - tabled, 'list looks table-driven after all'


def test_provider_without_a_table_entry_follows_the_default_convention():
    """A provider aisuite adds later must work immediately."""
    new = next(
        p
        for p in srv.PROVIDERS
        if p not in srv._KEY_FIELDS
        and p not in srv._LOCAL
        and p not in srv._OWN_CREDENTIALS
    )
    with fake_keyring({(new, 'api-key'): 'secret'}):
        config, error = srv.provider_config(new)
    assert error is None
    assert config == {'api_key': 'secret'}


def test_provider_without_a_table_entry_reports_the_missing_key_cleanly():
    """No KeyError on a provider we don't know about."""
    new = next(
        p
        for p in srv.PROVIDERS
        if p not in srv._KEY_FIELDS
        and p not in srv._LOCAL
        and p not in srv._OWN_CREDENTIALS
    )
    with fake_keyring({}):
        config, error = srv.provider_config(new)
    assert config is None
    assert f'keyring set {new} api-key' in error


@pytest.mark.parametrize('provider', ['aws', 'google'])
def test_own_credential_chain_asks_nothing_of_the_keyring(provider):
    """aws and google handle their own login; blocking them would be wrong."""
    with mock.patch.object(
        srv.keyring, 'get_password', side_effect=AssertionError('must not be called')
    ):
        config, error = srv.provider_config(provider)
    assert error is None
    assert config == {}


def test_optional_only_refers_to_existing_fields():
    for provider, fields in srv._OPTIONAL.items():
        known = srv._KEY_FIELDS.get(provider, srv._DEFAULT_FIELDS)
        unknown = fields - known.keys()
        assert not unknown, f'{provider}: unknown optional field {unknown}'


def test_no_provider_is_fully_optional():
    """Otherwise a provider with no key at all would count as "fine"."""
    for provider, fields in srv._OPTIONAL.items():
        known = srv._KEY_FIELDS.get(provider, srv._DEFAULT_FIELDS)
        assert fields != known.keys()


def test_tables_do_not_overlap():
    for a, b in [
        (srv._KEY_FIELDS.keys(), srv._LOCAL.keys()),
        (srv._KEY_FIELDS.keys(), srv._OWN_CREDENTIALS),
        (srv._LOCAL.keys(), srv._OWN_CREDENTIALS),
    ]:
        assert not (a & b), f'provider in two tables: {a & b}'


# --------------------------------------------------------------------------
# Against the real MCP protocol
# --------------------------------------------------------------------------


def _send(process, message):
    process.stdin.write(json.dumps(message) + '\n')
    process.stdin.flush()


def _receive(process):
    line = process.stdout.readline()
    return json.loads(line) if line else None


@pytest.fixture
def server_process():
    """Start the server as a separate process, the way Claude Desktop does."""
    process = subprocess.Popen(
        [sys.executable, str(Path(__file__).with_name('aisuite_mcp_server.py'))],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    timer = threading.Timer(30.0, process.kill)
    timer.start()
    try:
        _send(
            process,
            {
                'jsonrpc': '2.0',
                'id': 1,
                'method': 'initialize',
                'params': {
                    'protocolVersion': '2024-11-05',
                    'capabilities': {},
                    'clientInfo': {'name': 'test', 'version': '0'},
                },
            },
        )
        response = _receive(process)
        _send(process, {'jsonrpc': '2.0', 'method': 'notifications/initialized'})
        yield process, response
    finally:
        timer.cancel()
        process.terminate()
        process.wait(timeout=10)


def test_handshake_reports_name_and_version(server_process):
    _, response = server_process
    info = response['result']['serverInfo']
    assert info['name'] == 'aisuite-mcp'
    assert info['version'] == srv.__version__


def test_tool_appears_in_the_list(server_process):
    process, _ = server_process
    _send(process, {'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list', 'params': {}})
    tools = _receive(process)['result']['tools']

    assert [t['name'] for t in tools] == ['ask_language_model']
    fields = tools[0]['inputSchema']['properties']
    assert {'provider', 'model', 'question'} <= fields.keys()


def test_unknown_provider_over_the_protocol(server_process):
    """A real tools/call, so the MCP layer itself is exercised too."""
    process, _ = server_process
    _send(
        process,
        {
            'jsonrpc': '2.0',
            'id': 3,
            'method': 'tools/call',
            'params': {
                'name': 'ask_language_model',
                'arguments': {'provider': 'nonsense', 'model': 'x', 'question': 'hi'},
            },
        },
    )
    response = _receive(process)
    text = response['result']['content'][0]['text']
    assert 'unknown provider' in text
