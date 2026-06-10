"""URL + token settings dialog, persisted via config.py."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
)

from ..config import ClientConfig


class SettingsDialog(QDialog):
    def __init__(self, config: ClientConfig, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self._url = QLineEdit(config.server_url)
        self._token = QLineEdit(config.token or "")
        self._token.setEchoMode(QLineEdit.EchoMode.Password)
        self._save_token = QCheckBox("Save token to disk")
        self._save_token.setChecked(config.save_token)

        form = QFormLayout(self)
        form.addRow("Server URL", self._url)
        form.addRow("API token", self._token)
        form.addRow(self._save_token)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def values(self) -> ClientConfig:
        token = self._token.text().strip() or None
        return ClientConfig(
            server_url=self._url.text().strip(),
            token=token,
            save_token=self._save_token.isChecked(),
        )
