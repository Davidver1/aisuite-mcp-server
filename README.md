# aisuite-mcp-server

A small MCP server that exposes [aisuite](https://github.com/andrewyng/aisuite)'s
multi-provider chat completions as a single tool for MCP clients such as
Claude Cowork.

## Status

This is a personal utility script, shared as reference material alongside
[an issue on the aisuite repo](https://github.com/andrewyng/aisuite/issues)
asking whether an MCP *server* companion (as opposed to aisuite's existing
MCP *client* support) would be of interest to the project. It is **not**
packaged or published, has had limited testing, and its docstrings, error
messages and tool descriptions are in Dutch. Treat it as a working proof of
concept, not a finished library.

## What it does

One MCP tool, `vraag_taalmodel` ("ask language model"), backed by
`aisuite.Client()`. It covers 18 of aisuite's chat-capable providers:

```
anthropic, azure, cerebras, cohere, deepseek, fireworks, groq,
huggingface, inception, lmstudio, mistral, nebius, ollama, openai,
sambanova, together, watsonx, xai
```

`aws` and `google` are intentionally left out — they rely on their own,
more involved credential flow (boto3's credential chain, and a service
account JSON file, respectively) rather than a single API key, so they
don't fit the simple secret-lookup pattern used here.

Credentials are read from the OS keyring (via the `keyring` package) per
provider, at call time — nothing is loaded eagerly, and a missing key
produces a clear error telling you which `keyring set <provider> ...`
command to run, instead of a raw SDK exception.

## Why a separate repo, not a PR to aisuite

aisuite's own `mcp` extra pins `mcp<2.0.0` (needed for its MCP *client*
feature). Building an MCP *server* requires `mcp>=2.0.0` (the `MCPServer`
class only exists from 2.0 onward). Bundling both into the same package
would create an internal version conflict between aisuite's own extras —
so this lives as a separate, thin wrapper instead.

## Running it

Install (unpinned — there's no known conflict between these packages
outside of aisuite's own `mcp<2.0` constraint, which this script
deliberately avoids by installing `mcp` directly rather than via
`aisuite[mcp]`):

```
pip install aisuite anthropic boto3 cerebras_cloud_sdk cohere \
    google-cloud-speech vertexai groq ibm-watsonx-ai mistralai \
    openai huggingface_hub requests mcp keyring
```

Set whichever provider keys you actually plan to use:

```
keyring set mistral api-key
keyring set azure endpoint
keyring set azure api-key
```

(`ollama` and `lmstudio` run locally and need no key.)

Start it (stdio, the transport Claude Desktop expects):

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

## License

MIT — see [LICENSE](LICENSE).
