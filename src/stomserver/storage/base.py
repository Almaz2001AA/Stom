"""Storage interface for binary objects (volumes, masks)."""

from __future__ import annotations

from abc import ABC, abstractmethod


class StorageKeyError(KeyError):
    """Raised when a requested storage key does not exist."""


class Storage(ABC):
    @abstractmethod
    def put(self, key: str, data: bytes) -> None: ...

    @abstractmethod
    def get(self, key: str) -> bytes: ...

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def delete(self, key: str) -> None: ...
