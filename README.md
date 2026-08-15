# aisuite-mcp-server

A small MCP server that exposes [aisuite](https://github.com/andrewyng/aisuite)'s
multi-provider chat completions as a single tool for MCP clients such as
Claude Desktop and Claude Code.

## Status

This is a personal utility script, shared as reference material alongside
[issue #402 on the aisuite repo](https://github.com/andrewyng/aisuite/issues/402)
asking whether an MCP *server* companion (as opposed to aisuite's existing
MCP *client* support) would be of interest to the project. It is **not**
packaged or published, and its docstrings, error messages and tool
descriptions are in Dutch. Treat it as a working proof of concept, not a
finished library.

## What it does

One MCP tool, `vraag_taalmodel` ("ask language model"), backed by
`aisuite.Client()`. It covers **every provider aisuite ships** — the list
comes from `ProviderFactory.get_supported_providers()` at import time, not
from a table in this file, so a provider added in a future aisuite release
works here without any change. As of aisuite 0.1.14 that is 21 providers.

Deliberately, this server does not police that list. If a provider can't do
chat (`deepgram`), or its credentials aren't set up (`google`), aisuite
itself says so clearly — and gatekeeping would only block providers that
would otherwise work, such as `aws` for anyone who already has a boto3
credential chain configured.

## Credentials

Secrets are read from the OS keyring (via the `keyring` package), keyed by
provider name, at call time. The convention is one entry named `api-key`:

```
keyring set mistral api-key
keyring set openai api-key
```

Exceptions, which the script knows about:

```
keyring set azure endpoint
keyring set azure api-key          # api-version is optional
keyring set watsonx service-url
keyring set watsonx api-key
keyring set watsonx project-id
```

(`huggingface` also uses `api-key`, but aisuite wants it under the config
name `token` — handled internally.)

Three groups need nothing from the keyring:

- `ollama` and `lmstudio` run locally.
- `aws` and `google` use their own credential chains (boto3's, and
  `GOOGLE_APPLICATION_CREDENTIALS` respectively); the script passes an
  empty config and lets them resolve it.

Secrets are passed to aisuite as per-provider **configuration**, not
exported into the process environment — so they are not inherited by any
subprocess a library might spawn. A missing secret produces a message
naming the exact `keyring set` command to run, and only for the fields
actually missing, rather than a raw SDK exception.

Two things worth knowing:

- Keyring entries are per OS user account, not per virtualenv. Any program
  running as the same user can read them.
- aisuite instantiates a provider once and caches it, so a secret rotated
  after first use is picked up only after a restart.

## Transport

stdio only, deliberately. This server has no authentication of any kind, so
exposing it over HTTP/SSE would let anyone who can reach the port run model
calls at the operator's expense.

## Why a separate repo, not a PR to aisuite

aisuite's own `mcp` extra pins `mcp<2.0.0` (needed for its MCP *client*
feature). Building an MCP *server* requires `mcp>=2.0.0` (the `MCPServer`
class only exists from 2.0 onward). Bundling both into the same package
would create an internal version conflict between aisuite's own extras —
so this lives as a separate, thin wrapper instead.

## Running it

Install into its own environment (note: `mcp` is installed directly rather
than via `aisuite[mcp]`, which would pin it below 2.0):

```
pip install aisuite anthropic boto3 cerebras_cloud_sdk cohere \
    google-cloud-speech vertexai groq ibm-watsonx-ai mistralai \
    openai huggingface_hub requests mcp keyring
```

Start it:

```
python aisuite_mcp_server.py
```

Then point an MCP client at it, e.g. in `claude_desktop_config.json`:

```json
"aisuite-mcp": {
  "command": "/path/to/venv/bin/python",
  "args": ["/path/to/aisuite_mcp_server.py"]
}
```

## Checking your setup

`controle.py` reports, for every provider aisuite knows, whether it is
usable — SDK installed, credentials present, chat supported — without
making a single API call:

```
python controle.py
```

Each provider gets one of four stamps: `GEREED` (ready), `SLEUTEL` (needs a
keyring entry), `PAKKET` (SDK not installed), `GEEN CHAT` (transcription
only). To make one real call, which does cost money on a paid provider:

```
python controle.py --vraag ollama:mistral-small:latest
python controle.py --vraag openai:gpt-5.5
```

## Tests

```
pip install pytest
pytest -v
```

31 tests, no network access or real credentials required — the keyring is
mocked and the aisuite client replaced with a double. Coverage includes the
credential-mapping conventions (including that a provider with no table
entry falls back to the standard one, so future aisuite providers work),
the error paths (missing secrets, unreachable keyring, provider failures),
assertions that neither the prompt nor the secret reaches the log, and a
real stdio MCP handshake against the server running as a subprocess.

## License

MIT — see [LICENSE](LICENSE).
