"""Controleer de opzet van aisuite_mcp_server zonder iets te laten kosten.

Loopt alle providers langs en meldt per stuk of hij bruikbaar is: staat de
sleutel in de sleutelbos, is het pakket geinstalleerd, en kan de provider
uberhaupt chatten. Er gaat geen enkele aanroep naar een taalmodel, dus dit
kost niets en is zo klaar.

    python controle.py

Wilt u wel een echte aanroep doen, dan kan dat er los achteraan -- een
model per keer, zodat u zelf bepaalt wat het mag kosten:

    python controle.py --vraag ollama:mistral-small:latest
    python controle.py --vraag openai:gpt-5.5
"""

from __future__ import annotations

import argparse
import importlib
import json

import aisuite_mcp_server as srv


def _providerstatus(provider: str) -> tuple[str, str]:
    """Geef (stempel, toelichting) voor een provider, zonder netwerk."""
    # 1. Is het pakket er dat deze provider nodig heeft?
    try:
        module = importlib.import_module(f'aisuite.providers.{provider}_provider')
    except ImportError as exc:
        ontbrekend = str(exc).replace("No module named ", '').strip("'\" ")
        return 'PAKKET', f'{ontbrekend} niet geinstalleerd'

    # 2. Kan deze provider chatten? (deepgram doet alleen transcriptie)
    klasse = getattr(module, f'{provider.capitalize()}Provider', None)
    if klasse is not None:
        bron = getattr(klasse.chat_completions_create, '__doc__', '') or ''
        if 'does not support chat' in bron.lower():
            return 'GEEN CHAT', 'alleen audio, geen chatmodel'

    # 3. Zijn de inloggegevens rond?
    config, fout = srv.provider_config(provider)
    if fout is not None:
        if 'sleutelbos niet bereikbaar' in fout:
            # De reden staat al bovenaan het overzicht; hier alleen kort.
            return 'SLEUTEL', 'sleutelbos onbruikbaar (zie boven)'
        # De melding is opgebouwd als "<wat mist>. Zet ... met:\n<commando>".
        # In dit overzicht past alleen het eerste deel.
        return 'SLEUTEL', fout.split('. Zet ')[0]

    if provider in srv._LOKAAL:
        return 'GEREED', 'lokaal, geen sleutel nodig'
    if provider in srv._EIGEN_INLOG:
        return 'GEREED', 'eigen inlogketen (pas bij aanroep te merken)'
    return 'GEREED', f'sleutel gevonden ({", ".join(sorted(config))})'


def _sleutelbos_werkt() -> str | None:
    """Geef de reden terug als de sleutelbos onbruikbaar is, anders None.

    Eenmalig controleren: anders herhaalt dezelfde lange melding zich bij
    elke provider en is het overzicht niet meer te lezen.
    """
    try:
        srv.keyring.get_password('aisuite-mcp-controle', 'bestaat-niet')
    except Exception as exc:
        return srv.kort_fout(exc)
    return None


def overzicht() -> int:
    stempels = {'GEREED': 0, 'SLEUTEL': 0, 'PAKKET': 0, 'GEEN CHAT': 0}

    print(f'aisuite_mcp_server {srv.__version__}')
    print(f'{len(srv.PROVIDERS)} providers volgens aisuite\n')

    kapot = _sleutelbos_werkt()
    if kapot:
        print(f'  LET OP: de sleutelbos zelf is niet bruikbaar:\n    {kapot}\n')
        print('  Geen enkele provider met een sleutel kan werken tot dat is opgelost.')
        print('  Op Linux vraagt dat meestal om een sleutelbos-dienst (gnome-keyring,')
        print('  KWallet) of het pakket "keyrings.alt".\n')

    for provider in srv.PROVIDERS:
        stempel, toelichting = _providerstatus(provider)
        stempels[stempel] += 1
        print(f'  {stempel:10} {provider:14} {toelichting}')

    print()
    print(
        ' | '.join(
            f'{naam.lower()}: {aantal}' for naam, aantal in stempels.items() if aantal
        )
    )
    print(
        '\nSLEUTEL betekent alleen dat er nog niets in de sleutelbos staat -- '
        'zet die\nalleen voor providers die u echt gaat gebruiken.'
    )
    return 0


def stel_vraag(doel: str) -> int:
    """Doe een echte aanroep. Dit kost geld bij een betaalde provider."""
    if ':' not in doel:
        print(
            'Geef provider:model, bijvoorbeeld ollama:mistral-small:latest '
            f"-- niet '{doel}'"
        )
        return 2

    provider, model = doel.split(':', 1)
    import aisuite as ai

    print(f'-> {provider}:{model}')
    uitvoer = json.loads(
        srv.voer_uit(ai.Client(), provider, model, 'Zeg hallo in een zin.')
    )
    if 'fout' in uitvoer:
        print(f'   MISLUKT: {uitvoer["fout"]}')
        return 1
    print(f'   {uitvoer["antwoord"]}')
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--vraag',
        metavar='PROVIDER:MODEL',
        help='doe een echte aanroep bij een model (kost geld bij een betaalde '
        'provider)',
    )
    args = parser.parse_args()

    if args.vraag:
        return stel_vraag(args.vraag)
    return overzicht()


if __name__ == '__main__':
    raise SystemExit(main())
