from fastapi import FastAPI
from fastapi.testclient import TestClient

from stomserver.api.errors import install_error_handlers


def _app_with_boom():
    app = FastAPI()
    install_error_handlers(app)

    @app.get("/boom")
    def boom():
        raise RuntimeError("kaboom")

    return app


def test_unhandled_exception_returns_uniform_500():
    client = TestClient(_app_with_boom(), raise_server_exceptions=False)
    r = client.get("/boom")
    assert r.status_code == 500
    body = r.json()
    assert body["code"] == 500
    assert "kaboom" not in body["detail"]  # no traceback / internal leak
