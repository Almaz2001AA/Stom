"""Back-compat re-export: labels now live in :mod:`stomengine`."""

from __future__ import annotations

from stomengine.labels import DENTALSEGMENTATOR_LABELS  # noqa: F401

__all__ = ["DENTALSEGMENTATOR_LABELS"]
