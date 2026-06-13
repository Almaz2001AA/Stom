"""Export segmentation labels as STL surface meshes (for CAD / 3D printing).

A :class:`~stomcore.mask.SegmentationMask` is a labelled voxel volume; this turns
a chosen label into a watertight triangle surface and writes a binary STL. The
mesher is intentionally dependency-free (pure NumPy): it builds the *voxel-exact*
boundary surface — the exposed faces of the foreground voxels — so the output is
faithful to the segmentation (faceted rather than smoothed). Vertices are mapped
through the mask's :class:`~stomcore.geometry.Geometry` into physical millimetres,
so the STL lines up with the scan in any slicer/CAD package.

Keeping this in ``stomcore`` (no Qt, no torch) makes it unit-testable and reusable
by both the desktop client and the engine/CLI.
"""

from __future__ import annotations

import re
import struct
from pathlib import Path

import numpy as np

from .geometry import Geometry
from .mask import SegmentationMask

# Per-face-direction geometry: the array-axis whose neighbour decides whether a
# voxel face is exposed, the shift direction, and the four corner offsets (in
# (x, y, z), units of one voxel, relative to the voxel centre) wound so the
# triangle normal points *outward*. See module tests for the winding checks.
_HALF = 0.5
_FACES = (
    # (axis, step, corner offsets CCW seen from outside)
    (2, +1, ((+_HALF, -_HALF, -_HALF), (+_HALF, +_HALF, -_HALF),
             (+_HALF, +_HALF, +_HALF), (+_HALF, -_HALF, +_HALF))),   # +x
    (2, -1, ((-_HALF, -_HALF, -_HALF), (-_HALF, -_HALF, +_HALF),
             (-_HALF, +_HALF, +_HALF), (-_HALF, +_HALF, -_HALF))),   # -x
    (1, +1, ((-_HALF, +_HALF, -_HALF), (-_HALF, +_HALF, +_HALF),
             (+_HALF, +_HALF, +_HALF), (+_HALF, +_HALF, -_HALF))),   # +y
    (1, -1, ((-_HALF, -_HALF, -_HALF), (+_HALF, -_HALF, -_HALF),
             (+_HALF, -_HALF, +_HALF), (-_HALF, -_HALF, +_HALF))),   # -y
    (0, +1, ((-_HALF, -_HALF, +_HALF), (+_HALF, -_HALF, +_HALF),
             (+_HALF, +_HALF, +_HALF), (-_HALF, +_HALF, +_HALF))),   # +z
    (0, -1, ((-_HALF, -_HALF, -_HALF), (-_HALF, +_HALF, -_HALF),
             (+_HALF, +_HALF, -_HALF), (+_HALF, -_HALF, -_HALF))),   # -z
)


def _exposed(solid: np.ndarray, axis: int, step: int) -> np.ndarray:
    """Foreground voxels whose neighbour along ``axis``/``step`` is background.

    Out-of-array neighbours count as background, so faces on the volume edge are
    emitted too.
    """
    neighbour = np.zeros_like(solid)
    src = [slice(None)] * 3
    dst = [slice(None)] * 3
    if step > 0:
        dst[axis] = slice(None, -1)
        src[axis] = slice(1, None)
    else:
        dst[axis] = slice(1, None)
        src[axis] = slice(None, -1)
    neighbour[tuple(dst)] = solid[tuple(src)]
    return solid & ~neighbour


def binary_to_mesh(solid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Voxel-exact surface of a boolean ``[z, y, x]`` volume.

    Returns ``(verts, faces)`` where ``verts`` is ``(N, 3)`` continuous voxel
    indices in ``(x, y, z)`` order (voxel centres at integer coordinates) and
    ``faces`` is ``(M, 3)`` int triangle indices. The mesh is a triangle soup
    (vertices not shared between faces) — valid for STL, which needs no
    connectivity.
    """
    solid = np.ascontiguousarray(solid, dtype=bool)
    verts_parts: list[np.ndarray] = []
    faces_parts: list[np.ndarray] = []
    base = 0
    for axis, step, offsets in _FACES:
        zyx = np.argwhere(_exposed(solid, axis, step))
        if zyx.size == 0:
            continue
        centres = zyx[:, ::-1].astype(np.float64)          # (m, 3) -> (x, y, z)
        corners = centres[:, None, :] + np.asarray(offsets)  # (m, 4, 3)
        m = corners.shape[0]
        verts_parts.append(corners.reshape(-1, 3))
        f0 = base + 4 * np.arange(m)
        faces_parts.append(np.stack([f0, f0 + 1, f0 + 2], axis=1))
        faces_parts.append(np.stack([f0, f0 + 2, f0 + 3], axis=1))
        base += 4 * m
    if not verts_parts:
        return np.empty((0, 3), np.float64), np.empty((0, 3), np.int64)
    return np.concatenate(verts_parts), np.concatenate(faces_parts)


def weld_mesh(verts: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Merge coincident vertices of a triangle soup into a shared-vertex mesh.

    :func:`binary_to_mesh` emits a soup (4 unshared vertices per face); welding
    builds the vertex connectivity that smoothing needs. Corner coordinates are
    exact half-integers, so rounding before the unique is just defensive.
    """
    if len(verts) == 0:
        return verts, faces
    key = np.round(np.asarray(verts, dtype=np.float64), 6)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    return uniq, inv[np.asarray(faces)]


def _unique_edges(faces: np.ndarray) -> np.ndarray:
    """Undirected unique edges ``(E, 2)`` of a shared-vertex triangle mesh."""
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    return np.unique(np.sort(e, axis=1), axis=0)


def smooth_taubin(
    verts: np.ndarray,
    faces: np.ndarray,
    *,
    iterations: int = 12,
    lamb: float = 0.5,
    mu: float = -0.53,
) -> np.ndarray:
    """Taubin (λ|μ) smoothing of a shared-vertex mesh; returns moved vertices.

    Each iteration is a Laplacian shrinking pass (``+lamb``) followed by an
    inflating pass (``+mu``, μ<0); the pair largely cancels volume loss, so a
    tooth keeps its size while the voxel stair-steps melt away. ``faces`` are
    unchanged — only vertex positions move, so the surface stays watertight.
    Operate in whatever space the caller passes (we use physical mm so the
    smoothing radius is isotropic regardless of slice spacing).
    """
    verts = np.asarray(verts, dtype=np.float64)
    if iterations <= 0 or len(verts) == 0 or np.asarray(faces).size == 0:
        return verts
    verts = verts.copy()
    edges = _unique_edges(np.asarray(faces))
    a, b = edges[:, 0], edges[:, 1]
    deg = np.zeros(len(verts))
    np.add.at(deg, a, 1.0)
    np.add.at(deg, b, 1.0)
    deg[deg == 0] = 1.0  # isolated vertices never move

    def _pass(v: np.ndarray, factor: float) -> np.ndarray:
        nbr_sum = np.zeros_like(v)
        np.add.at(nbr_sum, a, v[b])
        np.add.at(nbr_sum, b, v[a])
        return v + factor * (nbr_sum / deg[:, None] - v)

    for _ in range(iterations):
        verts = _pass(verts, lamb)
        verts = _pass(verts, mu)
    return verts


def index_to_world(verts_xyz: np.ndarray, geometry: Geometry) -> np.ndarray:
    """Map continuous ``(x, y, z)`` voxel indices to physical mm (ITK convention).

    ``world = origin + direction @ (spacing * index)`` — the same mapping
    SimpleITK uses, so STL coordinates match the NIfTI/DICOM frame.
    """
    spacing = np.asarray(geometry.spacing, dtype=np.float64)
    origin = np.asarray(geometry.origin, dtype=np.float64)
    direction = np.asarray(geometry.direction, dtype=np.float64).reshape(3, 3)
    scaled = np.asarray(verts_xyz, dtype=np.float64) * spacing
    return scaled @ direction.T + origin


def _orient_outward(verts: np.ndarray, faces: np.ndarray, geometry: Geometry) -> np.ndarray:
    """Flip winding if the geometry's direction is left-handed (det < 0).

    The corner offsets wind outward in index space; a left-handed direction
    matrix mirrors world space and would invert every normal, so we swap two
    vertices of each triangle to keep normals pointing out of the surface.
    """
    direction = np.asarray(geometry.direction, dtype=np.float64).reshape(3, 3)
    if np.linalg.det(direction) < 0 and faces.size:
        faces = faces[:, [0, 2, 1]]
    return faces


def _triangle_normals(tris: np.ndarray) -> np.ndarray:
    """Unit normals for ``(M, 3, 3)`` triangles; degenerate faces get a zero normal."""
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    lengths = np.linalg.norm(n, axis=1, keepdims=True)
    return np.divide(n, lengths, out=np.zeros_like(n), where=lengths > 0)


def write_binary_stl(path: str | Path, verts_world: np.ndarray, faces: np.ndarray) -> None:
    """Write a binary STL: 80-byte header, triangle count, then 50 bytes/triangle."""
    verts_world = np.asarray(verts_world, dtype=np.float64)
    faces = np.asarray(faces)
    tris = verts_world[faces] if faces.size else np.empty((0, 3, 3))
    normals = _triangle_normals(tris)

    record = np.dtype([
        ("normal", "<f4", 3),
        ("v0", "<f4", 3),
        ("v1", "<f4", 3),
        ("v2", "<f4", 3),
        ("attr", "<u2"),
    ])
    assert record.itemsize == 50  # STL requires exactly 50 bytes per facet
    rows = np.zeros(len(faces), dtype=record)
    if faces.size:
        rows["normal"] = normals
        rows["v0"] = tris[:, 0]
        rows["v1"] = tris[:, 1]
        rows["v2"] = tris[:, 2]

    with open(path, "wb") as f:
        f.write(b"Stom STL export".ljust(80, b"\0"))
        f.write(struct.pack("<I", len(faces)))
        f.write(rows.tobytes())


def label_to_stl(
    mask: SegmentationMask,
    label_id: int,
    path: str | Path,
    *,
    smooth_iterations: int = 0,
) -> int:
    """Write label ``label_id`` of ``mask`` to ``path`` as STL; return triangle count.

    The label is cropped to its bounding box before meshing so a single tooth in
    a full-CBCT mask costs only its own voxels, not the whole volume.
    ``smooth_iterations`` > 0 welds the surface and applies that many Taubin
    passes (in mm) to round off the voxel stair-steps; 0 keeps it voxel-exact.
    """
    solid = mask.labels == label_id
    located = np.argwhere(solid)
    if located.size == 0:
        write_binary_stl(path, np.empty((0, 3)), np.empty((0, 3), np.int64))
        return 0
    lo = located.min(axis=0)
    hi = located.max(axis=0) + 1
    sub = solid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]

    verts, faces = binary_to_mesh(sub)
    verts += lo[::-1]  # crop offset, (z, y, x) lo -> (x, y, z) vertex frame
    if smooth_iterations > 0:
        verts, faces = weld_mesh(verts, faces)
    faces = _orient_outward(verts, faces, mask.geometry)
    world = index_to_world(verts, mask.geometry)
    if smooth_iterations > 0:
        world = smooth_taubin(world, faces, iterations=smooth_iterations)
    write_binary_stl(path, world, faces)
    return len(faces)


def _safe_filename(label_id: int, name: str) -> str:
    slug = re.sub(r"[^0-9A-Za-zА-Яа-яЁё]+", "_", name).strip("_") or "label"
    return f"{label_id:02d}_{slug}.stl"


def export_labels_stl(
    mask: SegmentationMask,
    out_dir: str | Path,
    label_ids: list[int] | None = None,
    *,
    smooth_iterations: int = 0,
    progress=None,
) -> list[Path]:
    """Write one STL per label into ``out_dir``; return the files written.

    ``label_ids`` defaults to every label actually present in the mask. Empty
    labels are skipped. ``smooth_iterations`` rounds off the voxel surface (see
    :func:`label_to_stl`). ``progress(done, total, name)`` is called per label so
    a UI can report which structure is being exported.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    present = mask.present_labels()
    ids = sorted(present if label_ids is None else (set(label_ids) & present))

    written: list[Path] = []
    for i, label_id in enumerate(ids):
        info = mask.label_map.get(label_id)
        name = info.name if info is not None else str(label_id)
        if progress is not None:
            progress(i, len(ids), name)
        dest = out_dir / _safe_filename(label_id, name)
        if label_to_stl(mask, label_id, dest, smooth_iterations=smooth_iterations) > 0:
            written.append(dest)
    if progress is not None:
        progress(len(ids), len(ids), "")
    return written
