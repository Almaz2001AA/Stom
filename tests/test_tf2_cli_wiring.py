"""Tests for ToothFairy2 model selection in the engine CLI."""

from pathlib import Path

from stomengine.cli import _build_engine, resolve_model_dir, selected_model
from stomengine.engine import InProcessEngine
from stomengine.labels import DENTALSEGMENTATOR_LABELS
from stomengine.runner import DentalSegmentatorRunner, ToothFairy2Runner
from stomengine.tf2_labels import TOOTHFAIRY2_LABELS


def test_selected_model_default_is_dentalsegmentator():
    assert selected_model({}) == "dentalsegmentator"
    assert selected_model({"STOM_MODEL": "other"}) == "dentalsegmentator"


def test_selected_model_toothfairy2_aliases():
    assert selected_model({"STOM_MODEL": "toothfairy2"}) == "toothfairy2"
    assert selected_model({"STOM_MODEL": "TF2"}) == "toothfairy2"
    assert selected_model({"STOM_MODEL": " ToothFairy2 "}) == "toothfairy2"


def test_resolve_model_dir_picks_tf2_subpath(monkeypatch):
    monkeypatch.delenv("STOM_MODEL_DIR", raising=False)
    tf2 = Path(resolve_model_dir("toothfairy2"))
    default = Path(resolve_model_dir("dentalsegmentator"))
    assert tf2.parts[-1] == "Dataset112_ToothFairy2"  # flat model dir
    assert default.parts[-2] == "Dataset112_DentalSegmentator_v100"


def test_resolve_model_dir_env_overrides_model(monkeypatch):
    monkeypatch.setenv("STOM_MODEL_DIR", "/custom/tf2")
    assert resolve_model_dir("toothfairy2") == "/custom/tf2"


def test_build_engine_default_uses_dentalsegmentator(monkeypatch):
    monkeypatch.delenv("STOM_ENGINE_FAKE", raising=False)
    monkeypatch.delenv("STOM_MODEL", raising=False)
    engine = _build_engine()
    assert isinstance(engine, InProcessEngine)
    assert isinstance(engine._runner, DentalSegmentatorRunner)
    assert engine._labels is DENTALSEGMENTATOR_LABELS


def test_build_engine_toothfairy2_uses_tf2_runner_and_labels(monkeypatch):
    monkeypatch.delenv("STOM_ENGINE_FAKE", raising=False)
    monkeypatch.setenv("STOM_MODEL", "toothfairy2")
    engine = _build_engine()
    assert isinstance(engine, InProcessEngine)
    assert isinstance(engine._runner, ToothFairy2Runner)
    assert engine._labels is TOOTHFAIRY2_LABELS
