"""Launch the Stom desktop client."""

from __future__ import annotations

import sys


def main() -> int:
    from PySide6.QtWidgets import QApplication

    from .app_controller import AppController
    from .cloud_client import CloudClient
    from .config import load
    from .ui.main_window import MainWindow

    config = load()
    cloud = CloudClient(config.server_url, config.token) if config.server_url else None

    app = QApplication(sys.argv)
    window = MainWindow(AppController(cloud_client=cloud))
    window.resize(1100, 800)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
