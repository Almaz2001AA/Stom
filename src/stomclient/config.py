"""Client configuration persisted to ~/.config/stom/client.toml."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

import tomli_w

DEFAULT_PATH = Path.home() / ".config" / "stom" / "client.toml"


@dataclass
class ClientConfig:
    server_url: str = ""
    token: str | None = None
    save_token: bool = True


def load(path: str | os.PathLike | None = None) -> ClientConfig:
    p = Path(path) if path is not None else DEFAULT_PATH
    if not p.exists():
        return ClientConfig()
    data = tomllib.loads(p.read_text())
    return ClientConfig(
        server_url=data.get("server_url", ""),
        token=data.get("token"),
        save_token=data.get("save_token", True),
    )


def save(config: ClientConfig, path: str | os.PathLike | None = None) -> None:
    p = Path(path) if path is not None else DEFAULT_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "server_url": config.server_url,
        "save_token": config.save_token,
    }
    if config.save_token and config.token:
        payload["token"] = config.token
    p.write_text(tomli_w.dumps(payload))
    os.chmod(p, 0o600)
