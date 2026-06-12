"""Launch the Stom desktop client."""

from __future__ import annotations

import sys


def main() -> int:
    from PySide6.QtWidgets import QApplication

    from .app_controller import AppController
    from .cloud_client import CloudClient
    from .config import load
    from .local_engine import build_local_engine
    from .ui.main_window import MainWindow

    config = load()
    cloud = CloudClient(config.server_url, config.token) if config.server_url else None
    engine = build_local_engine()

    app = QApplication(sys.argv)
    window = MainWindow(AppController(cloud_client=cloud, engine=engine))
    window.resize(1100, 800)
    window.show()
    # Fail-soft engine + client update checks, after the window is up.
    window.check_for_updates()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
