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
    smooth_taubin,
    weld_mesh,
    write_binary_stl,
)


def _sphere_mask(n=24, r=8):
    zz, yy, xx = np.mgrid[0:n, 0:n, 0:n]
    c = n / 2
    labels = np.zeros((n, n, n), dtype=np.uint8)
    labels[(xx - c) ** 2 + (yy - c) ** 2 + (zz - c) ** 2 < r ** 2] = 1
    return labels


def _edge_share_counts(faces):
    counts = {}
    for tri in faces:
        a, b, c = tri
        for u, w in ((a, b), (b, c), (c, a)):
            counts[(min(u, w), max(u, w))] = counts.get((min(u, w), max(u, w)), 0) + 1
    return counts


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


def test_weld_mesh_shares_vertices_and_keeps_topology():
    verts, faces = binary_to_mesh(_solid_cube(2))
    wv, wf = weld_mesh(verts, faces)
    assert len(wv) < len(verts)                       # coincident corners merged
    assert wf.shape == faces.shape
    # Welded surface is watertight: every edge shared by exactly two triangles.
    counts = _edge_share_counts(wf)
    assert counts and all(v == 2 for v in counts.values())
    # Indices stay in range and the rounded coordinate set is unchanged.
    assert wf.max() < len(wv)
    assert {tuple(np.round(v, 3)) for v in wv} == {tuple(np.round(v, 3)) for v in verts}


def test_smooth_reduces_facet_normal_variance_but_keeps_watertight():
    verts, faces = weld_mesh(*binary_to_mesh(_sphere_mask()))

    def normal_spread(v):
        tris = v[faces]
        n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        return float(np.var(n, axis=0).sum())

    smoothed = smooth_taubin(verts.astype(float), faces, iterations=12)
    assert smoothed.shape == verts.shape
    assert normal_spread(smoothed) < normal_spread(verts.astype(float))  # rounder
    # Faces untouched -> still watertight after smoothing.
    counts = _edge_share_counts(faces)
    assert all(v == 2 for v in counts.values())


def test_smoothing_preserves_overall_size_no_collapse():
    # Taubin must not shrink the sphere to a point (pure Laplacian would).
    verts, faces = weld_mesh(*binary_to_mesh(_sphere_mask(n=24, r=8)))
    verts = verts.astype(float)
    before = verts.max(axis=0) - verts.min(axis=0)
    after_v = smooth_taubin(verts, faces, iterations=12)
    after = after_v.max(axis=0) - after_v.min(axis=0)
    assert np.all(after > before * 0.8)               # keeps >80% of its extent


def test_smooth_iterations_zero_is_voxel_exact(tmp_path):
    mask = SegmentationMask(_sphere_mask().astype(np.uint8),
                            Geometry.identity((0.4, 0.4, 0.4)),
                            {1: LabelInfo(1, "tooth", (1, 1, 1))})
    raw = tmp_path / "raw.stl"
    label_to_stl(mask, 1, raw, smooth_iterations=0)
    verts, _ = binary_to_mesh(_sphere_mask().astype(bool))
    count, _ = _read_stl(raw)
    assert count == len(binary_to_mesh(_sphere_mask().astype(bool))[1])  # unchanged soup


def test_label_to_stl_smoothing_moves_vertices(tmp_path):
    mask = SegmentationMask(_sphere_mask().astype(np.uint8),
                            Geometry.identity((0.4, 0.4, 0.4)),
                            {1: LabelInfo(1, "tooth", (1, 1, 1))})
    raw, smooth = tmp_path / "raw.stl", tmp_path / "smooth.stl"
    label_to_stl(mask, 1, raw, smooth_iterations=0)
    label_to_stl(mask, 1, smooth, smooth_iterations=12)
    _, raw_tris = _read_stl(raw)
    _, smooth_tris = _read_stl(smooth)
    raw_pts = np.array([p for t in raw_tris for p in t])
    smooth_pts = np.array([p for t in smooth_tris for p in t])
    # Both describe the same sphere but the smoothed surface is geometrically different.
    assert not np.allclose(np.sort(raw_pts, axis=0)[:100],
                           np.sort(smooth_pts, axis=0)[:100])


def test_cyrillic_label_name_yields_usable_filename(tmp_path):
    labels = np.zeros((3, 3, 3), dtype=np.uint8)
    labels[1, 1, 1] = 2
    mask = SegmentationMask(labels, Geometry.identity(spacing=(1, 1, 1)),
                            {2: LabelInfo(2, "Нижняя челюсть", (0, 0, 0))})
    written = export_labels_stl(mask, tmp_path)
    assert written and written[0].name.startswith("02_")
    assert written[0].name.endswith(".stl")
