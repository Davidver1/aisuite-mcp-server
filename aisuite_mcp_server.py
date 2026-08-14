"""MCP-server die aisuite's taalmodellen aanbiedt als gereedschap voor Claude.

Achttien van aisuite's twintig chatproviders staan hier achter een enkel
gereedschap, vraag_taalmodel:

    anthropic, azure, cerebras, cohere, deepseek, fireworks, groq,
    huggingface, inception, lmstudio, mistral, nebius, ollama, openai,
    sambanova, together, watsonx, xai

aws en google ontbreken bewust: die leunen op een eigen, uitgebreidere
inlogketen (boto3's credential-keten, resp. een dienstaccountbestand) in
plaats van een enkele tekstsleutel, en passen niet in het sleutelbos-patroon
hieronder.

De sleutels komen uit dezelfde Windows-sleutelbos die de rest van de
Python-omgeving al gebruikt, bijvoorbeeld:

    keyring set mistral api-key
    keyring set azure endpoint
    keyring set azure api-key

ollama en lmstudio draaien lokaal en hebben geen sleutel nodig. anthropic en
openai gebruiken de sleutels die al voor de lab-venv gezet zijn (dezelfde
sleutelbos, systeembreed) -- daar hoeft niets bij.

Draait het beste in een eigen omgeving, los van een eventuele bestaande
aisuite-installatie: aisuite's eigen "mcp"-extra pint mcp<2.0.0 (voor de
MCP-clientfunctie), terwijl deze server mcp>=2.0.0 nodig heeft voor de
MCPServer-klasse. Beide in dezelfde omgeving geeft een importfout.

Starten (stdio, de vorm die Claude Desktop verwacht):

    python aisuite_mcp_server.py

Installeren:

    pip install aisuite anthropic boto3 cerebras_cloud_sdk cohere \\
        google-cloud-speech vertexai groq ibm-watsonx-ai mistralai \\
        openai huggingface_hub requests mcp keyring
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Annotated, Any

try:
    import keyring
except ImportError as exc:  # pragma: no cover - alleen bij een kapotte installatie
    raise SystemExit(
        'Deze server heeft het pakket "keyring" nodig. Installeer het met:\n'
        '    pip install keyring'
    ) from exc

try:
    from pydantic import Field
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        'Deze server heeft het pakket "pydantic" nodig (komt normaal mee met "mcp").'
    ) from exc

try:
    from mcp.server import MCPServer
except ImportError as exc:  # pragma: no cover - alleen zonder de afhankelijkheid
    raise SystemExit(
        'Deze server heeft het pakket "mcp" nodig. Installeer het met:\n'
        '    pip install mcp'
    ) from exc

try:
    import aisuite as ai
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        'Deze server heeft het pakket "aisuite" nodig. Installeer het met:\n'
        '    pip install aisuite'
    ) from exc

LOG = logging.getLogger(__name__)

# Providers die helemaal geen sleutel nodig hebben: ze draaien lokaal.
_GEEN_SLEUTEL_NODIG = frozenset({'ollama', 'lmstudio'})

# provider -> omgevingsvariabele, voor providers met een enkele sleutel.
# Elke sleutel staat in de sleutelbos onder service=<provider>, naam="api-key".
_ENV_PER_PROVIDER = {
    'anthropic': 'ANTHROPIC_API_KEY',
    'cerebras': 'CEREBRAS_API_KEY',
    'cohere': 'CO_API_KEY',
    'deepseek': 'DEEPSEEK_API_KEY',
    'fireworks': 'FIREWORKS_API_KEY',
    'groq': 'GROQ_API_KEY',
    'huggingface': 'HF_TOKEN',
    'inception': 'INCEPTION_API_KEY',
    'mistral': 'MISTRAL_API_KEY',
    'nebius': 'NEBIUS_API_KEY',
    'openai': 'OPENAI_API_KEY',
    'sambanova': 'SAMBANOVA_API_KEY',
    'together': 'TOGETHER_API_KEY',
    'xai': 'XAI_API_KEY',
}

# provider -> {sleutelbos-veldnaam: omgevingsvariabele}, voor providers met
# meer dan een geheim. De sleutelbos-service is de providernaam, net als
# hierboven; alleen de gebruikersnaam varieert per veld.
_MULTI_ENV_PER_PROVIDER = {
    'azure': {
        'endpoint': 'AZURE_BASE_URL',
        'api-key': 'AZURE_API_KEY',
        'api-version': 'AZURE_API_VERSION',
    },
    'watsonx': {
        'service-url': 'WATSONX_SERVICE_URL',
        'api-key': 'WATSONX_API_KEY',
        'project-id': 'WATSONX_PROJECT_ID',
    },
}
_MULTI_OPTIONEEL = {'azure': {'api-version'}}

PROVIDERS = sorted(
    _ENV_PER_PROVIDER.keys() | _MULTI_ENV_PER_PROVIDER.keys() | _GEEN_SLEUTEL_NODIG
)

INSTRUCTIES = """\
Dit gereedschap stuurt een chatvraag naar een taalmodel bij een van de
ondersteunde providers en geeft het antwoord terug. Kies de provider en de
modelnaam zoals de provider zelf ze noemt (bijvoorbeeld "llama-3.3-70b" bij
together, of "gpt-oss:20b" bij ollama) -- dit gereedschap kent zelf geen
modelnamen en geeft ze ongewijzigd door.

ollama en lmstudio draaien lokaal op deze machine en hebben geen sleutel
nodig; alle andere providers wel. Ontbreekt een sleutel, dan meldt het
gereedschap dat met het commando om ze te zetten, in plaats van de
onbegrijpelijke fout die de provider zelf zou geven.
"""


def _resultaat(waarde: dict[str, Any]) -> str:
    """Geef een resultaat als compacte JSON terug."""
    return json.dumps(waarde, ensure_ascii=False, indent=2, default=str)


def _zet_sleutel(provider: str) -> str | None:
    """Haal de sleutel van *provider* uit de sleutelbos en zet 'm in de omgeving.

    Geeft een foutmelding terug (in plaats van 'm te gooien) als de sleutel
    ontbreekt of de sleutelbos zelf niet bereikbaar is, zodat de aanroeper
    die netjes als gereedschapsresultaat kan tonen in plaats van de server
    te laten crashen.
    """
    if provider in _GEEN_SLEUTEL_NODIG:
        return None

    if provider in _MULTI_ENV_PER_PROVIDER:
        velden = _MULTI_ENV_PER_PROVIDER[provider]
        optioneel = _MULTI_OPTIONEEL.get(provider, set())
        try:
            waarden = {veld: keyring.get_password(provider, veld) for veld in velden}
        except Exception as exc:  # bijv. geen sleutelbos-backend beschikbaar
            return f"sleutelbos niet bereikbaar voor '{provider}': {exc}"

        ontbreekt = [v for v in velden if v not in optioneel and not waarden.get(v)]
        if ontbreekt:
            regels = '\n'.join(f'    keyring set {provider} {v}' for v in velden)
            return (
                f"{provider} mist {', '.join(ontbreekt)} in de sleutelbos. Zet ze met:\n"
                f'{regels}'
            )
        for veld, omgevingsvariabele in velden.items():
            if waarden.get(veld):
                os.environ[omgevingsvariabele] = waarden[veld]
        return None

    try:
        sleutel = keyring.get_password(provider, 'api-key')
    except Exception as exc:  # bijv. geen sleutelbos-backend beschikbaar
        return f"sleutelbos niet bereikbaar voor '{provider}': {exc}"

    if not sleutel:
        return (
            f"geen sleutel gevonden voor '{provider}'. Zet 'm met:\n"
            f"    keyring set {provider} api-key"
        )
    os.environ[_ENV_PER_PROVIDER[provider]] = sleutel
    return None


def bouw_server() -> MCPServer:
    # ollama laadt een groot model bij de eerste (koude) aanroep vanaf schijf
    # in het geheugen; de standaard van 30s is daarvoor te krap. Zie ook de
    # OLLAMA_KEEP_ALIVE-instelling, die bepaalt hoe lang een model daarna
    # warm blijft en dus hoe vaak dit erna nog een rol speelt.
    client = ai.Client(provider_configs={'ollama': {'timeout': 300}})
    server = MCPServer(
        name='aisuite-mcp',
        title='Taalmodellen zonder eigen SDK',
        version='0.1.0',
        instructions=INSTRUCTIES,
    )

    @server.tool(
        name='vraag_taalmodel',
        description=(
            'Stuur een chatvraag naar een taalmodel bij een van de '
            'ondersteunde providers (' + ', '.join(PROVIDERS) + ') en geef '
            'het antwoord terug.'
        ),
    )
    def vraag_taalmodel(
        provider: Annotated[
            str, Field(description=f"Een van: {', '.join(PROVIDERS)}")
        ],
        model: Annotated[
            str, Field(description='Modelnaam zoals de provider zelf die noemt.')
        ],
        vraag: Annotated[str, Field(description='De vraag of opdracht.')],
        systeeminstructie: Annotated[
            str | None, Field(description='Optionele systeemboodschap.')
        ] = None,
        temperature: Annotated[
            float | None, Field(description='0.0-2.0, hoger is willekeuriger.')
        ] = None,
    ) -> str:
        if provider not in PROVIDERS:
            return _resultaat(
                {'fout': f"onbekende provider '{provider}'. Kies uit: {', '.join(PROVIDERS)}"}
            )

        fout = _zet_sleutel(provider)
        if fout:
            return _resultaat({'fout': fout})

        berichten: list[dict[str, str]] = []
        if systeeminstructie:
            berichten.append({'role': 'system', 'content': systeeminstructie})
        berichten.append({'role': 'user', 'content': vraag})

        kwargs: dict[str, Any] = {}
        if temperature is not None:
            kwargs['temperature'] = temperature

        try:
            antwoord = client.chat.completions.create(
                model=f'{provider}:{model}',
                messages=berichten,
                **kwargs,
            )
        except Exception as exc:  # de providers gooien niet allemaal dezelfde klasse
            LOG.warning('%s:%s mislukt: %s', provider, model, exc)
            return _resultaat({'fout': str(exc)})

        return _resultaat({'antwoord': antwoord.choices[0].message.content})

    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog='aisuite-mcp',
        description=(
            'MCP-server die taalmodellen zonder eigen SDK aanbiedt als '
            'gereedschap voor Claude, via aisuite.'
        ),
    )
    parser.add_argument(
        '--transport',
        default='stdio',
        choices=['stdio', 'streamable-http', 'sse'],
        help='hoe de server bereikbaar is (standaard: stdio, voor Claude Desktop)',
    )
    parser.add_argument(
        '--log',
        default='warning',
        choices=['debug', 'info', 'warning', 'error'],
        help='hoeveel logregels naar stderr',
    )
    args = parser.parse_args(argv)

    # Logregels moeten naar stderr: stdout draagt bij stdio het MCP-protocol.
    logging.basicConfig(
        level=getattr(logging, args.log.upper(), logging.WARNING),
        format='%(asctime)s %(levelname)-7s %(name)s: %(message)s',
        stream=sys.stderr,
    )

    bouw_server().run(transport=args.transport)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
