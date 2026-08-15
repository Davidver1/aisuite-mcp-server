# Changelog

Notable changes to this project, loosely following
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/). Versions before
0.3.0 predate this file.

## 0.4.0

Breaking change: the code switches from Dutch to English throughout,
which renames the tool and its parameters.

- The MCP tool is now `ask_language_model` (was `vraag_taalmodel`), with
  parameters `question` and `system_prompt` (were `vraag` and
  `systeeminstructie`). The JSON result keys are now `error`/`answer`
  (were `fout`/`antwoord`). Existing MCP client configuration doesn't
  need to change for this, since it only points at the script's file
  path, not at the tool name.
- `controle.py` is renamed to `check_setup.py`; its `--vraag` flag is now
  `--ask`, and its status stamps are `READY`/`KEY`/`PACKAGE`/`NO CHAT`
  (were `GEREED`/`SLEUTEL`/`PAKKET`/`GEEN CHAT`).
- `provider_config()` now returns a `ConfigError` (a `str` subclass
  carrying `.summary` and `.keyring_unreachable`) instead of a plain
  message string, so `check_setup.py` no longer has to parse the message
  text to tell a missing key apart from an unreachable keyring backend.
- Added a GitHub Actions workflow that runs the test suite on
  Python 3.11-3.13 for every push and pull request.
- Added `requirements.txt` as the single source for the core
  dependencies (`aisuite`, `mcp>=2.0.0`, `keyring`). The module docstring
  and README no longer each spell out the full list of every provider
  SDK; providers load lazily, so only the SDK for a provider actually in
  use is needed, and `check_setup.py` reports which one is still
  missing.

## 0.3.0

- The provider list is now read from aisuite
  (`ProviderFactory.get_supported_providers()`) instead of a hand-written
  table, so a provider aisuite adds later works here immediately.
- Secrets are passed to aisuite via `client.configure()` instead of
  `os.environ`, so a child process a library starts can no longer
  inherit them.
