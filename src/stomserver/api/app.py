"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI

from ..config import Config, load_config
from ..queue.rq_queue import RqJobQueue
from ..storage.base import Storage
from ..storage.local import LocalFileStorage
from .errors import install_error_handlers


def create_app(
    config: Config | None = None,
    *,
    session_factory=None,
    storage: Storage | None = None,
    queue=None,
) -> FastAPI:
    cfg = config or load_config()
    app = FastAPI(title="stomserver")

    if session_factory is None:
        from ..db.session import create_all, make_engine, make_session_factory

        engine = make_engine(cfg.db_url)
        create_all(engine)
        session_factory = make_session_factory(engine)
    app.state.session_factory = session_factory
    app.state.storage = storage or LocalFileStorage(cfg.storage_dir)

    if queue is None:
        import redis

        queue = RqJobQueue(redis.Redis.from_url(cfg.redis_url))
    app.state.queue = queue
    app.state.config = cfg

    install_error_handlers(app)

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    from .routes_jobs import router as jobs_router
    from .routes_studies import router as studies_router

    app.include_router(studies_router)
    app.include_router(jobs_router)
    return app
