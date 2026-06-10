import pytest

from stomcore.geometry import Geometry
from stomclient import slice_renderer as sr
from stomclient.measurement import LinearMeasurement, MeasurementSet, plane_spacing


def test_plane_spacing_maps_axes():
    geo = Geometry.identity(spacing=(0.3, 0.4, 0.5))  # x, y, z
    assert plane_spacing(geo, sr.AXIAL) == (0.3, 0.4)     # col=x, row=y
    assert plane_spacing(geo, sr.CORONAL) == (0.3, 0.5)   # col=x, row=z
    assert plane_spacing(geo, sr.SAGITTAL) == (0.4, 0.5)  # col=y, row=z


def test_length_isotropic_vertical():
    geo = Geometry.identity(spacing=(0.3, 0.3, 0.3))
    m = LinearMeasurement(p0=(0.0, 0.0), p1=(0.0, 10.0), plane=sr.AXIAL, geometry=geo)
    assert m.length_mm == pytest.approx(3.0)


def test_length_anisotropic():
    geo = Geometry.identity(spacing=(0.5, 1.0, 2.0))
    # axial: col spacing 0.5, row spacing 1.0; 3 cols, 4 rows
    m = LinearMeasurement(p0=(0.0, 0.0), p1=(3.0, 4.0), plane=sr.AXIAL, geometry=geo)
    # dx=3*0.5=1.5, dy=4*1.0=4.0 -> hypot=4.272...
    assert m.length_mm == pytest.approx(4.27200187, rel=1e-6)


def test_measurement_set_add_and_clear():
    geo = Geometry.identity(spacing=(1, 1, 1))
    ms = MeasurementSet()
    ms.add(LinearMeasurement((0, 0), (0, 1), sr.AXIAL, geo))
    assert len(ms) == 1
    ms.clear()
    assert len(ms) == 0
