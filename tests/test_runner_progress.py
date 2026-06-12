"""Progress reporting: the tqdm shim that turns nnU-Net's tile loop into a
(done, total) callback, and the FakeRunner's progress wiring."""

import contextlib

import numpy as np
import pytest

from stomcore.geometry import Geometry
from stomcore.volume import Volume
from stomengine.runner import FakeRunner, _tile_progress


def _volume():
    return Volume(np.zeros((4, 5, 6), dtype=np.int16), Geometry.identity((0.3, 0.3, 0.3)))


def test_tile_progress_is_noop_without_callback():
    assert isinstance(_tile_progress(None), contextlib.nullcontext)


def test_fake_runner_reports_a_step():
    seen = []
    FakeRunner().predict(_volume(), progress=lambda d, t: seen.append((d, t)))
    assert seen == [(1, 1)]


def test_tile_progress_routes_nnunet_tqdm_to_callback():
    """Inside the context, nnU-Net's tqdm symbol becomes our reporting shim and
    is restored afterward."""
    pr = pytest.importorskip("nnunetv2.inference.predict_from_raw_data")
    original = pr.tqdm

    seen = []
    with _tile_progress(lambda d, t: seen.append((d, t))):
        # Mimic nnU-Net's usage: tqdm(total=n) as a context manager + update().
        with pr.tqdm(desc=None, total=3, disable=True) as pbar:
            pbar.update()
            pbar.update()
            pbar.update()

    assert pr.tqdm is original          # symbol restored
    assert seen[0] == (0, 3)            # reports immediately when the bar opens
    assert seen[-1] == (3, 3)          # and after the final tile
    assert (2, 3) in seen
