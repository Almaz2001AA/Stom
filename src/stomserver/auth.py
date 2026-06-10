"""Token hashing and the request auth dependency."""

from __future__ import annotations

import hashlib

from fastapi import Depends, Header, HTTPException

from .api.deps import get_db
from .db.models import Account, ApiToken

# RFC 7235: 401 responses to a Bearer-protected resource must advertise the scheme.
_UNAUTHORIZED_HEADERS = {"WWW-Authenticate": "Bearer"}


def hash_token(token: str) -> str:
    """SHA-256 hex digest of a raw token. Only the hash is ever stored."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def get_current_account(
    authorization: str | None = Header(default=None),
    db=Depends(get_db),
) -> Account:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token",
                            headers=_UNAUTHORIZED_HEADERS)
    token = authorization[len("Bearer "):]
    row = db.query(ApiToken).filter_by(token_hash=hash_token(token)).first()
    if row is None:
        raise HTTPException(status_code=401, detail="invalid token",
                            headers=_UNAUTHORIZED_HEADERS)
    account = db.get(Account, row.account_id)
    if account is None:
        raise HTTPException(status_code=401, detail="invalid token",
                            headers=_UNAUTHORIZED_HEADERS)
    return account
