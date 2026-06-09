"""Token hashing and (later) the request auth dependency."""

from __future__ import annotations

import hashlib


def hash_token(token: str) -> str:
    """SHA-256 hex digest of a raw token. Only the hash is ever stored."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()
