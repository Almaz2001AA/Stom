import struct

import numpy as np
import pytest

from stomcore.geometry import Geometry
from stomcore.mask import LabelInfo, SegmentationMask
from stomcore.stl_export import (
    binary_to_mesh,
    export_labels_stl,
    index_to_world,
    label_to_stl,
    write_binary_stl,
)


def _solid_cube(n: int = 1) -> np.ndarray:
    v = np.zeros((n + 2, n + 2, n + 2), dtype=bool)
    v[1:-1, 1:-1, 1:-1] = True
    return v


def _read_stl(path):
    data = path.read_bytes()
    (count,) = struct.unpack("<I", data[80:84])
    tris = []
    off = 84
    for _ in range(count):
        vals = struct.unpack("<12f", data[off:off + 48])
        tris.append((vals[3:6], vals[6:9], vals[9:12]))
        off += 50
    return count, tris


def test_single_voxel_is_a_closed_box():
    verts, faces = binary_to_mesh(np.ones((1, 1, 1), dtype=bool))
    assert faces.shape == (12, 3)        # 6 faces * 2 triangles
    assert verts.shape == (24, 3)        # triangle soup: 4 verts per face


def test_mesh_is_watertight_each_edge_shared_twice():
    # A watertight closed surface: every undirected edge belongs to exactly two
    # triangles. Dedupe coincident soup vertices first.
    verts, faces = binary_to_mesh(_solid_cube(3))
    keys = {tuple(np.round(v, 3)) for v in verts}
    lookup = {k: i for i, k in enumerate(keys)}
    vid = np.array([lookup[tuple(np.round(v, 3))] for v in verts])
    edge_count: dict[tuple[int, int], int] = {}
    for tri in faces:
        a, b, c = vid[tri]
        for u, w in ((a, b), (b, c), (c, a)):
            edge_count[(min(u, w), max(u, w))] = edge_count.get((min(u, w), max(u, w)), 0) + 1
    assert edge_count and all(v == 2 for v in edge_count.values())


def test_face_normals_point_outward_for_single_voxel(tmp_path):
    # Each facet of a single voxel centred at the origin must point away from it.
    mask = SegmentationMask(
        np.ones((1, 1, 1), dtype=np.uint8),
        Geometry.identity(spacing=(1.0, 1.0, 1.0)),
        {1: LabelInfo(1, "x", (255, 0, 0))},
    )
    path = tmp_path / "v.stl"
    label_to_stl(mask, 1, path)
    _, tris = _read_stl(path)
    centre = np.array([0.0, 0.0, 0.0])
    for v0, v1, v2 in tris:
        tri = np.array([v0, v1, v2])
        n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        outward = tri.mean(axis=0) - centre
        assert np.dot(n, outward) > 0     # normal agrees with outward direction


def test_index_to_world_applies_spacing_origin_direction():
    geo = Geometry(spacing=(0.5, 0.5, 2.0), origin=(10.0, 20.0, 30.0),
                   direction=(1, 0, 0, 0, 1, 0, 0, 0, 1))
    world = index_to_world(np.array([[2.0, 4.0, 3.0]]), geo)
    assert np.allclose(world[0], [10 + 1.0, 20 + 2.0, 30 + 6.0])


def test_world_size_matches_spacing(tmp_path):
    # A 2x2x2-voxel block at 0.4mm spacing spans 2*0.4 = 0.8mm per axis.
    labels = np.zeros((4, 4, 4), dtype=np.uint8)
    labels[1:3, 1:3, 1:3] = 1
    mask = SegmentationMask(labels, Geometry.identity(spacing=(0.4, 0.4, 0.4)),
                            {1: LabelInfo(1, "block", (1, 2, 3))})
    path = tmp_path / "b.stl"
    label_to_stl(mask, 1, path)
    _, tris = _read_stl(path)
    pts = np.array([p for tri in tris for p in tri])
    extent = pts.max(axis=0) - pts.min(axis=0)
    assert np.allclose(extent, [0.8, 0.8, 0.8])


def test_left_handed_direction_keeps_normals_outward(tmp_path):
    # A mirrored (det < 0) direction must not invert the surface.
    mask = SegmentationMask(
        np.ones((1, 1, 1), dtype=np.uint8),
        Geometry(spacing=(1, 1, 1), origin=(0, 0, 0),
                 direction=(-1, 0, 0, 0, 1, 0, 0, 0, 1)),
        {1: LabelInfo(1, "m", (0, 0, 0))},
    )
    path = tmp_path / "m.stl"
    label_to_stl(mask, 1, path)
    _, tris = _read_stl(path)
    centre = np.array([0.0, 0.0, 0.0])
    for v0, v1, v2 in tris:
        tri = np.array([v0, v1, v2])
        n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        assert np.dot(n, tri.mean(axis=0) - centre) > 0


def test_binary_stl_byte_layout(tmp_path):
    verts, faces = binary_to_mesh(np.ones((1, 1, 1), dtype=bool))
    world = verts.astype(float)
    path = tmp_path / "x.stl"
    write_binary_stl(path, world, faces)
    size = path.stat().st_size
    assert size == 84 + 12 * 50          # header+count + 12 facets
    count, _ = _read_stl(path)
    assert count == 12


def test_label_to_stl_empty_label_writes_zero_triangles(tmp_path):
    mask = SegmentationMask(np.zeros((3, 3, 3), dtype=np.uint8),
                            Geometry.identity(spacing=(1, 1, 1)),
                            {1: LabelInfo(1, "absent", (0, 0, 0))})
    path = tmp_path / "empty.stl"
    assert label_to_stl(mask, 1, path) == 0
    count, _ = _read_stl(path)
    assert count == 0


def test_export_labels_one_file_per_present_label(tmp_path):
    labels = np.zeros((5, 5, 5), dtype=np.uint8)
    labels[1, 1, 1] = 3
    labels[3, 3, 3] = 4
    mask = SegmentationMask(
        labels, Geometry.identity(spacing=(1, 1, 1)),
        {3: LabelInfo(3, "Upper Teeth", (1, 1, 1)),
         4: LabelInfo(4, "Lower Teeth", (2, 2, 2)),
         9: LabelInfo(9, "Unused", (3, 3, 3))},   # not present -> skipped
    )
    seen = []
    written = export_labels_stl(mask, tmp_path, progress=lambda d, t, n: seen.append((d, t)))
    names = sorted(p.name for p in written)
    assert names == ["03_Upper_Teeth.stl", "04_Lower_Teeth.stl"]
    assert all(p.exists() for p in written)
    assert seen[-1] == (2, 2)            # final progress hits total


def test_export_labels_respects_explicit_id_filter(tmp_path):
    labels = np.zeros((5, 5, 5), dtype=np.uint8)
    labels[1, 1, 1] = 3
    labels[3, 3, 3] = 4
    mask = SegmentationMask(
        labels, Geometry.identity(spacing=(1, 1, 1)),
        {3: LabelInfo(3, "A", (1, 1, 1)), 4: LabelInfo(4, "B", (2, 2, 2))},
    )
    written = export_labels_stl(mask, tmp_path, label_ids=[4])
    assert [p.name for p in written] == ["04_B.stl"]


def test_cyrillic_label_name_yields_usable_filename(tmp_path):
    labels = np.zeros((3, 3, 3), dtype=np.uint8)
    labels[1, 1, 1] = 2
    mask = SegmentationMask(labels, Geometry.identity(spacing=(1, 1, 1)),
                            {2: LabelInfo(2, "Нижняя челюсть", (0, 0, 0))})
    written = export_labels_stl(mask, tmp_path)
    assert written and written[0].name.startswith("02_")
    assert written[0].name.endswith(".stl")
