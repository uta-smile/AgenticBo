import pytest

from sara.client import SaraClient


def test_local_server_environment_and_timeout(monkeypatch):
    monkeypatch.setenv("SARA_BASE_URL","http://127.0.0.1:8080/v1")
    monkeypatch.setenv("SARA_MODEL","qwen3.5:9b")
    monkeypatch.setenv("SARA_TIMEOUT_SECONDS","600")
    client=SaraClient.from_environment()
    assert client.model=="qwen3.5:9b" and client.timeout==600
    for invalid in ("nan","0","-1"):
        monkeypatch.setenv("SARA_TIMEOUT_SECONDS",invalid)
        with pytest.raises(ValueError,match="timeout"):
            SaraClient.from_environment()
