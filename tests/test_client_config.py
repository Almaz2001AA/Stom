import stat

from stomclient.config import ClientConfig, load, save


def test_load_missing_returns_defaults(tmp_path):
    cfg = load(tmp_path / "nope.toml")
    assert cfg == ClientConfig(server_url="", token=None, save_token=True)


def test_save_then_load_roundtrip(tmp_path):
    path = tmp_path / "client.toml"
    save(ClientConfig(server_url="https://api.example", token="abc"), path)
    cfg = load(path)
    assert cfg.server_url == "https://api.example"
    assert cfg.token == "abc"


def test_token_not_persisted_when_save_token_false(tmp_path):
    path = tmp_path / "client.toml"
    save(ClientConfig(server_url="u", token="secret", save_token=False), path)
    assert "secret" not in path.read_text()
    assert load(path).token is None


def test_saved_file_is_owner_only(tmp_path):
    path = tmp_path / "client.toml"
    save(ClientConfig(server_url="u", token="t"), path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
