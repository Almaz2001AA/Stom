"""Admin CLI: create an Account and issue an API token (printed once)."""

from __future__ import annotations

import argparse
import secrets

from stomserver.auth import hash_token
from stomserver.config import load_config
from stomserver.db.models import Account, ApiToken
from stomserver.db.session import create_all, make_engine, make_session_factory


def create_account(session_factory, name: str) -> str:
    """Create an account + token; return the RAW token (store it, it's shown once)."""
    db = session_factory()
    try:
        account = Account(name=name)
        db.add(account)
        db.flush()
        token = secrets.token_urlsafe(32)
        db.add(ApiToken(token_hash=hash_token(token), account_id=account.id))
        db.commit()
        return token
    finally:
        db.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create an account and issue an API token.")
    parser.add_argument("name", help="account / clinic name")
    args = parser.parse_args(argv)

    cfg = load_config()
    engine = make_engine(cfg.db_url)
    create_all(engine)
    factory = make_session_factory(engine)
    token = create_account(factory, args.name)
    print(f"account '{args.name}' created. API token (store it now, shown once):")
    print(token)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
