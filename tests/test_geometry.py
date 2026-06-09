import pytest

from stomcore.geometry import Geometry

IDENTITY_DIR = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def test_identity_builds_expected_geometry():
    g = Geometry.identity(spacing=(0.3, 0.3, 0.3))
    assert g.spacing == (0.3, 0.3, 0.3)
    assert g.origin == (0.0, 0.0, 0.0)
    assert g.direction == IDENTITY_DIR


def test_geometry_is_frozen():
    g = Geometry.identity(spacing=(0.3, 0.3, 0.3))
    with pytest.raises(Exception):
        g.spacing = (1.0, 1.0, 1.0)


def test_compatible_when_within_tolerance():
    a = Geometry(spacing=(0.30000, 0.3, 0.3), origin=(0, 0, 0), direction=IDENTITY_DIR)
    b = Geometry(spacing=(0.30001, 0.3, 0.3), origin=(0, 0, 0), direction=IDENTITY_DIR)
    assert a.is_compatible(b, tol=1e-3) is True


def test_incompatible_when_spacing_differs_beyond_tolerance():
    a = Geometry.identity(spacing=(0.3, 0.3, 0.3))
    b = Geometry.identity(spacing=(0.4, 0.3, 0.3))
    assert a.is_compatible(b) is False


def test_incompatible_when_origin_differs():
    a = Geometry.identity(spacing=(0.3, 0.3, 0.3))
    b = Geometry(spacing=(0.3, 0.3, 0.3), origin=(5.0, 0.0, 0.0), direction=IDENTITY_DIR)
    assert a.is_compatible(b) is False


def test_rejects_wrong_length_direction():
    with pytest.raises(ValueError):
        Geometry(spacing=(0.3, 0.3, 0.3), origin=(0, 0, 0), direction=(1.0, 0.0, 0.0))
