"""Request-scoped dependencies, backed by objects on app.state."""

from __future__ import annotations

from collections.abc import Iterator

from fastapi import Request
from sqlalchemy.orm import Session

from ..storage.base import Storage


def get_db(request: Request) -> Iterator[Session]:
    session = request.app.state.session_factory()
    try:
        yield session
    finally:
        session.close()


def get_storage(request: Request) -> Storage:
    return request.app.state.storage


def get_queue(request: Request):
    return request.app.state.queue
