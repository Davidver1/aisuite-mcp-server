"""Check the setup of aisuite_mcp_server without anything costing money.

Walks through every provider and reports whether it's usable: is the key
in the keyring, is the package installed, and can the provider even do
chat. No call ever goes out to a language model, so this costs nothing
and is done immediately.

    python check_setup.py

Want to make a real call too? That's separate and comes after -- one
model at a time, so you decide what it's allowed to cost:

    python check_setup.py --ask ollama:mistral-small:latest
    python check_setup.py --ask openai:gpt-5.5
"""

from __future__ import annotations

import argparse
import importlib
import json

import aisuite_mcp_server as srv


def _provider_status(provider: str) -> tuple[str, str]:
    """Return (stamp, note) for a provider, without any network access."""
    # 1. Is the package this provider needs installed?
    try:
        module = importlib.import_module(f'aisuite.providers.{provider}_provider')
    except ImportError as exc:
        missing = str(exc).replace("No module named ", '').strip("'\" ")
        return 'PACKAGE', f'{missing} not installed'

    # 2. Can this provider do chat? (deepgram only does transcription)
    cls = getattr(module, f'{provider.capitalize()}Provider', None)
    if cls is not None:
        doc = getattr(cls.chat_completions_create, '__doc__', '') or ''
        if 'does not support chat' in doc.lower():
            return 'NO CHAT', 'audio only, no chat model'

    # 3. Are the credentials in order?
    config, error = srv.provider_config(provider)
    if error is not None:
        if 'keyring unreachable' in error:
            # The reason is already at the top of the overview; keep it short here.
            return 'KEY', 'keyring unusable (see above)'
        # The message is built as "<what's missing>. Set ... with:\n<command>".
        # Only the first part fits in this overview.
        return 'KEY', error.split('. Set ')[0]

    if provider in srv._LOCAL:
        return 'READY', 'local, no key needed'
    if provider in srv._OWN_CREDENTIALS:
        return 'READY', 'own credential chain (only shows up at call time)'
    return 'READY', f'key found ({", ".join(sorted(config))})'


def _keyring_unreachable_reason() -> str | None:
    """Return the reason if the keyring is unusable, otherwise None.

    Checked once: otherwise the same long message would repeat for every
    provider and the overview would stop being readable.
    """
    try:
        srv.keyring.get_password('aisuite-mcp-check', 'does-not-exist')
    except Exception as exc:
        return srv.short_error(exc)
    return None


def overview() -> int:
    stamps = {'READY': 0, 'KEY': 0, 'PACKAGE': 0, 'NO CHAT': 0}

    print(f'aisuite_mcp_server {srv.__version__}')
    print(f'{len(srv.PROVIDERS)} providers according to aisuite\n')

    broken = _keyring_unreachable_reason()
    if broken:
        print(f'  NOTE: the keyring itself is not usable:\n    {broken}\n')
        print('  No provider that needs a key can work until that is fixed.')
        print('  On Linux this usually needs a keyring service (gnome-keyring,')
        print('  KWallet) or the "keyrings.alt" package.\n')

    for provider in srv.PROVIDERS:
        stamp, note = _provider_status(provider)
        stamps[stamp] += 1
        print(f'  {stamp:10} {provider:14} {note}')

    print()
    print(
        ' | '.join(
            f'{name.lower()}: {count}' for name, count in stamps.items() if count
        )
    )
    print(
        "\nKEY only means nothing is in the keyring yet -- set it only for "
        'providers you actually plan to use.'
    )
    return 0


def ask_question(target: str) -> int:
    """Make a real call. This costs money on a paid provider."""
    if ':' not in target:
        print(
            'Give provider:model, for example ollama:mistral-small:latest '
            f"-- not '{target}'"
        )
        return 2

    provider, model = target.split(':', 1)
    import aisuite as ai

    print(f'-> {provider}:{model}')
    output = json.loads(
        srv.run_query(ai.Client(), provider, model, 'Say hello in one sentence.')
    )
    if 'error' in output:
        print(f'   FAILED: {output["error"]}')
        return 1
    print(f'   {output["answer"]}')
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--ask',
        metavar='PROVIDER:MODEL',
        help='make a real call to a model (costs money on a paid provider)',
    )
    args = parser.parse_args()

    if args.ask:
        return ask_question(args.ask)
    return overview()


if __name__ == '__main__':
    raise SystemExit(main())
