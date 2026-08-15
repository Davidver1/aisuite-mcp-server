"""Tests voor aisuite_mcp_server.

Draaien:

    pip install pytest
    pytest -v

De tests raken geen enkele provider en hebben dus geen netwerk of sleutels
nodig: de sleutelbos wordt nagebootst en de aisuite-client vervangen door
een dubbelganger die opschrijft waarmee hij is aangeroepen.
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
# Hulpmiddelen
# --------------------------------------------------------------------------


class NepAntwoord:
    """Bootst het antwoordobject van aisuite na."""

    def __init__(self, inhoud: str):
        bericht = type('Bericht', (), {'content': inhoud})()
        keuze = type('Keuze', (), {'message': bericht})()
        self.choices = [keuze]


class NepClient:
    """Client-dubbelganger die onthoudt waarmee hij is aangeroepen."""

    def __init__(self, antwoord: str = 'hallo', fout: Exception | None = None):
        self.configuraties: list[dict] = []
        self.aanroepen: list[dict] = []
        self._antwoord = antwoord
        self._fout = fout
        zelf = self

        class Completions:
            def create(self, **kwargs):
                zelf.aanroepen.append(kwargs)
                if zelf._fout:
                    raise zelf._fout
                return NepAntwoord(zelf._antwoord)

        class Chat:
            completions = Completions()

        self.chat = Chat()

    def configure(self, configs: dict) -> None:
        self.configuraties.append(configs)


def nep_sleutelbos(waarden: dict[tuple[str, str], str | None]):
    """Vervang keyring.get_password door een opzoeking in *waarden*."""
    return mock.patch.object(
        srv.keyring, 'get_password', side_effect=lambda s, u: waarden.get((s, u))
    )


# --------------------------------------------------------------------------
# provider_config
# --------------------------------------------------------------------------


def test_lokale_provider_heeft_geen_sleutel_nodig():
    config, fout = srv.provider_config('ollama')
    assert fout is None
    assert config == {'timeout': 300}


def test_lokale_config_is_een_kopie():
    """Aanpassen van het resultaat mag de module-tabel niet raken."""
    config, _ = srv.provider_config('ollama')
    config['timeout'] = 1
    opnieuw, _ = srv.provider_config('ollama')
    assert opnieuw['timeout'] == 300


def test_enkele_sleutel_wordt_omgezet_naar_aisuite_naam():
    with nep_sleutelbos({('mistral', 'api-key'): 'geheim'}):
        config, fout = srv.provider_config('mistral')
    assert fout is None
    assert config == {'api_key': 'geheim'}


def test_huggingface_gebruikt_token_in_plaats_van_api_key():
    """De sleutelbos-naam is overal gelijk; aisuite wil hier "token"."""
    with nep_sleutelbos({('huggingface', 'api-key'): 'hf_abc'}):
        config, fout = srv.provider_config('huggingface')
    assert fout is None
    assert config == {'token': 'hf_abc'}


def test_ontbrekende_sleutel_geeft_melding_met_commando():
    with nep_sleutelbos({}):
        config, fout = srv.provider_config('mistral')
    assert config is None
    assert 'keyring set mistral api-key' in fout


def test_meervoudig_noemt_alleen_de_ontbrekende_velden():
    """Velden die al goed staan horen niet in de reparatie-instructie."""
    with nep_sleutelbos({('watsonx', 'api-key'): 'geheim'}):
        config, fout = srv.provider_config('watsonx')
    assert config is None
    assert 'keyring set watsonx service-url' in fout
    assert 'keyring set watsonx project-id' in fout
    assert 'keyring set watsonx api-key' not in fout


def test_meervoudig_compleet_levert_alle_config_sleutels():
    with nep_sleutelbos(
        {
            ('watsonx', 'service-url'): 'https://eu-de.ml.cloud.ibm.com',
            ('watsonx', 'api-key'): 'geheim',
            ('watsonx', 'project-id'): 'p-123',
        }
    ):
        config, fout = srv.provider_config('watsonx')
    assert fout is None
    assert config == {
        'service_url': 'https://eu-de.ml.cloud.ibm.com',
        'api_key': 'geheim',
        'project_id': 'p-123',
    }


def test_optioneel_veld_mag_ontbreken():
    """azure werkt zonder api-version; die staat in _OPTIONEEL."""
    with nep_sleutelbos(
        {('azure', 'endpoint'): 'https://x.azure.com', ('azure', 'api-key'): 'geheim'}
    ):
        config, fout = srv.provider_config('azure')
    assert fout is None
    assert config == {'base_url': 'https://x.azure.com', 'api_key': 'geheim'}


def test_onbereikbare_sleutelbos_geeft_melding_geen_uitzondering():
    with mock.patch.object(
        srv.keyring, 'get_password', side_effect=RuntimeError('geen backend')
    ):
        config, fout = srv.provider_config('mistral')
    assert config is None
    assert 'sleutelbos niet bereikbaar' in fout


# --------------------------------------------------------------------------
# voer_uit
# --------------------------------------------------------------------------


def test_geslaagde_aanroep_geeft_het_antwoord():
    client = NepClient(antwoord='Hallo daar.')
    with nep_sleutelbos({('mistral', 'api-key'): 'geheim'}):
        uitvoer = json.loads(srv.voer_uit(client, 'mistral', 'small', 'hoi'))
    assert uitvoer == {'antwoord': 'Hallo daar.'}
    assert client.aanroepen[0]['model'] == 'mistral:small'
    assert client.aanroepen[0]['messages'] == [{'role': 'user', 'content': 'hoi'}]


def test_sleutel_gaat_via_configure_en_niet_via_de_omgeving(monkeypatch):
    """De kern van de veiligheidskeuze: niets in os.environ."""
    monkeypatch.delenv('MISTRAL_API_KEY', raising=False)
    client = NepClient()
    with nep_sleutelbos({('mistral', 'api-key'): 'geheim'}):
        srv.voer_uit(client, 'mistral', 'small', 'hoi')

    assert client.configuraties == [{'mistral': {'api_key': 'geheim'}}]
    import os

    assert 'MISTRAL_API_KEY' not in os.environ


def test_systeeminstructie_komt_vooraan():
    client = NepClient()
    with nep_sleutelbos({('mistral', 'api-key'): 'geheim'}):
        srv.voer_uit(client, 'mistral', 'small', 'hoi', systeeminstructie='wees kort')
    assert client.aanroepen[0]['messages'][0] == {
        'role': 'system',
        'content': 'wees kort',
    }


def test_temperature_wordt_alleen_meegestuurd_als_hij_gezet_is():
    client = NepClient()
    with nep_sleutelbos({('mistral', 'api-key'): 'geheim'}):
        srv.voer_uit(client, 'mistral', 'small', 'hoi')
        assert 'temperature' not in client.aanroepen[0]

        srv.voer_uit(client, 'mistral', 'small', 'hoi', temperature=0.5)
        assert client.aanroepen[1]['temperature'] == 0.5


def test_onbekende_provider_wordt_afgevangen_voor_de_sleutelbos():
    client = NepClient()
    uitvoer = json.loads(srv.voer_uit(client, 'onzin', 'x', 'hoi'))
    assert 'onbekende provider' in uitvoer['fout']
    assert client.aanroepen == []


def test_fout_van_provider_wordt_een_melding_geen_crash():
    client = NepClient(fout=RuntimeError('verbinding geweigerd'))
    with nep_sleutelbos({('mistral', 'api-key'): 'geheim'}):
        uitvoer = json.loads(srv.voer_uit(client, 'mistral', 'small', 'hoi'))
    assert uitvoer['fout'] == 'verbinding geweigerd'


def test_lange_foutmelding_wordt_afgekapt():
    client = NepClient(fout=RuntimeError('x' * 2000))
    with nep_sleutelbos({('mistral', 'api-key'): 'geheim'}):
        uitvoer = json.loads(srv.voer_uit(client, 'mistral', 'small', 'hoi'))
    assert len(uitvoer['fout']) < 600
    assert uitvoer['fout'].endswith('[...afgekapt]')


def test_ontbrekende_sleutel_bereikt_de_provider_niet():
    client = NepClient()
    with nep_sleutelbos({}):
        uitvoer = json.loads(srv.voer_uit(client, 'mistral', 'small', 'hoi'))
    assert 'keyring set' in uitvoer['fout']
    assert client.aanroepen == []


def test_de_vraag_komt_niet_in_het_logboek(caplog):
    """Wat de gebruiker vraagt hoort niet in stderr terecht te komen."""
    client = NepClient(fout=RuntimeError('kapot'))
    with nep_sleutelbos({('mistral', 'api-key'): 'geheim'}):
        srv.voer_uit(client, 'mistral', 'small', 'staat bedrijf X failliet?')
    assert 'failliet' not in caplog.text


def test_de_sleutel_komt_niet_in_het_logboek(caplog):
    client = NepClient(fout=RuntimeError('kapot'))
    with nep_sleutelbos({('mistral', 'api-key'): 'zeergeheim123'}):
        srv.voer_uit(client, 'mistral', 'small', 'hoi')
    assert 'zeergeheim123' not in caplog.text


# --------------------------------------------------------------------------
# Tabellen
# --------------------------------------------------------------------------


def test_providerlijst_komt_van_aisuite_niet_van_onszelf():
    """De server hoort niet te bepalen welke providers bestaan."""
    from aisuite.provider import ProviderFactory

    assert set(srv.PROVIDERS) == set(ProviderFactory.get_supported_providers())


def test_lijst_bevat_providers_die_in_geen_enkele_tabel_staan():
    """Bewijst dat de lijst niet stiekem uit de tabellen wordt afgeleid."""
    getabelleerd = srv._SLEUTELVELDEN.keys() | srv._LOKAAL.keys() | srv._EIGEN_INLOG
    assert set(srv.PROVIDERS) - getabelleerd, 'lijst lijkt toch tabelgedreven'


def test_provider_zonder_tabelregel_volgt_de_standaardconventie():
    """Een provider die aisuite later toevoegt moet meteen werken."""
    nieuw = next(
        p
        for p in srv.PROVIDERS
        if p not in srv._SLEUTELVELDEN
        and p not in srv._LOKAAL
        and p not in srv._EIGEN_INLOG
    )
    with nep_sleutelbos({(nieuw, 'api-key'): 'geheim'}):
        config, fout = srv.provider_config(nieuw)
    assert fout is None
    assert config == {'api_key': 'geheim'}


def test_provider_zonder_tabelregel_meldt_netjes_dat_de_sleutel_mist():
    """Geen KeyError op een provider die wij niet kennen."""
    nieuw = next(
        p
        for p in srv.PROVIDERS
        if p not in srv._SLEUTELVELDEN
        and p not in srv._LOKAAL
        and p not in srv._EIGEN_INLOG
    )
    with nep_sleutelbos({}):
        config, fout = srv.provider_config(nieuw)
    assert config is None
    assert f'keyring set {nieuw} api-key' in fout


@pytest.mark.parametrize('provider', ['aws', 'google'])
def test_eigen_inlogketen_vraagt_niets_aan_de_sleutelbos(provider):
    """aws en google regelen hun eigen inlog; blokkeren zou onterecht zijn."""
    with mock.patch.object(
        srv.keyring, 'get_password', side_effect=AssertionError('niet aanroepen')
    ):
        config, fout = srv.provider_config(provider)
    assert fout is None
    assert config == {}


def test_optioneel_verwijst_alleen_naar_bestaande_velden():
    for provider, velden in srv._OPTIONEEL.items():
        bekend = srv._SLEUTELVELDEN.get(provider, srv._STANDAARD_VELDEN)
        onbekend = velden - bekend.keys()
        assert not onbekend, f'{provider}: onbekend optioneel veld {onbekend}'


def test_geen_provider_is_volledig_optioneel():
    """Anders zou een provider zonder enige sleutel als "in orde" gelden."""
    for provider, velden in srv._OPTIONEEL.items():
        bekend = srv._SLEUTELVELDEN.get(provider, srv._STANDAARD_VELDEN)
        assert velden != bekend.keys()


def test_tabellen_overlappen_niet():
    for a, b in [
        (srv._SLEUTELVELDEN.keys(), srv._LOKAAL.keys()),
        (srv._SLEUTELVELDEN.keys(), srv._EIGEN_INLOG),
        (srv._LOKAAL.keys(), srv._EIGEN_INLOG),
    ]:
        assert not (a & b), f'provider in twee tabellen: {a & b}'


# --------------------------------------------------------------------------
# Over het echte MCP-protocol
# --------------------------------------------------------------------------


def _stuur(proces, bericht):
    proces.stdin.write(json.dumps(bericht) + '\n')
    proces.stdin.flush()


def _ontvang(proces):
    regel = proces.stdout.readline()
    return json.loads(regel) if regel else None


@pytest.fixture
def server_proces():
    """Start de server als los proces, zoals Claude Desktop dat doet."""
    proces = subprocess.Popen(
        [sys.executable, str(Path(__file__).with_name('aisuite_mcp_server.py'))],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    wekker = threading.Timer(30.0, proces.kill)
    wekker.start()
    try:
        _stuur(
            proces,
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
        antwoord = _ontvang(proces)
        _stuur(proces, {'jsonrpc': '2.0', 'method': 'notifications/initialized'})
        yield proces, antwoord
    finally:
        wekker.cancel()
        proces.terminate()
        proces.wait(timeout=10)


def test_handshake_meldt_naam_en_versie(server_proces):
    _, antwoord = server_proces
    info = antwoord['result']['serverInfo']
    assert info['name'] == 'aisuite-mcp'
    assert info['version'] == srv.__version__


def test_gereedschap_staat_in_de_lijst(server_proces):
    proces, _ = server_proces
    _stuur(proces, {'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list', 'params': {}})
    gereedschappen = _ontvang(proces)['result']['tools']

    assert [g['name'] for g in gereedschappen] == ['vraag_taalmodel']
    velden = gereedschappen[0]['inputSchema']['properties']
    assert {'provider', 'model', 'vraag'} <= velden.keys()


def test_onbekende_provider_over_het_protocol(server_proces):
    """Een echte tools/call, zodat ook de MCP-laag zelf getest is."""
    proces, _ = server_proces
    _stuur(
        proces,
        {
            'jsonrpc': '2.0',
            'id': 3,
            'method': 'tools/call',
            'params': {
                'name': 'vraag_taalmodel',
                'arguments': {'provider': 'onzin', 'model': 'x', 'vraag': 'hoi'},
            },
        },
    )
    antwoord = _ontvang(proces)
    tekst = antwoord['result']['content'][0]['text']
    assert 'onbekende provider' in tekst
