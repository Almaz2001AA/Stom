"""Memory-frugal nnU-Net inference + export for large multi-class models.

The 49-class ToothFairy2 model produces a ``(49, Z, Y, X)`` float16 logits
buffer (~13 GB at 0.3 mm full-head). nnU-Net's default export then upcasts that
to float32 to resample the probabilities (~26 GB) before argmax — which OOMs the
6-8 GB consumer PCs we target (three full-scan runs died with SIGKILL/no
traceback). This module keeps peak RAM low in two complementary ways:

1. **Disk-backed logits buffer.** The sliding-window accumulator is a
   ``numpy.memmap`` rather than a resident tensor, so the OS pages cold Z-slabs
   out to disk instead of holding all ~13 GB in RAM. Only the working set (the
   tiles touched recently) stays resident.
2. **Label-space export.** We argmax the logits on the network grid in Z-slabs
   (producing a 1-byte ``uint8`` label volume, never the float32 copy), then
   nearest-resample the *labels* — one channel, not 49 channels of float32 —
   back onto the input geometry, reusing nnU-Net's own crop/transpose
   bookkeeping so the result lines up exactly with the original scan.

The only deviation from nnU-Net's reference export is argmax-then-nearest-resample
instead of resample-then-argmax. For the 0.3 mm -> finer upsampling this differs
only by sub-voxel boundary jitter, which is immaterial for the segmentation
product and far cheaper in memory.

The sliding-window override is copied from nnunetv2 2.8.0
(``_internal_predict_sliding_window_return_logits``); it is pinned to that
version, which the frozen engine-pack bundles. If nnU-Net is upgraded, re-diff
the upstream method against ``_internal_predict_sliding_window_return_logits``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

import numpy as np


_TRUTHY_VALUES = {"1", "true", "yes", "on"}

# Z-slab depth for the chunked argmax. 32 slices of a 49x533x533 plane is
# ~0.9 GB float16 resident per slab — small enough for an 8 GB PC, large enough
# that the per-slab Python overhead is negligible.
DEFAULT_ARGMAX_SLAB_Z = 32


def low_memory_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Whether to use the disk-backed memmap inference path.

    On by default for the memory-heavy ToothFairy2 model; set
    ``STOM_DISABLE_LOWMEM`` to a truthy value to force the in-RAM path (faster
    when RAM is plentiful, but OOMs the full 49-class scan on small machines).
    """
    env = os.environ if env is None else env
    return env.get("STOM_DISABLE_LOWMEM", "").strip().lower() not in _TRUTHY_VALUES


def _nearest_resample_labels(
    labels: np.ndarray, target_shape: tuple[int, ...]
) -> np.ndarray:
    """Nearest-neighbour resample a label volume to ``target_shape``.

    Operates on the 1-channel ``uint8`` label array (not 49-channel logits), so
    it is cheap in memory. Uses an explicit floor index map rather than
    ``scipy.ndimage.zoom`` so the output shape is exactly ``target_shape`` (zoom
    can be off by one from rounding) and no interpolation/one-hot blow-up
    occurs.
    """
    src = labels.shape
    if tuple(int(s) for s in src) == tuple(int(s) for s in target_shape):
        return labels
    assert labels.ndim == len(target_shape), "rank mismatch in label resample"
    index = [
        np.minimum((np.arange(t) * s / t).astype(np.intp), s - 1)
        for s, t in zip(src, target_shape, strict=True)
    ]
    return labels[np.ix_(*index)]


def chunked_argmax_z(logits, slab_z: int = DEFAULT_ARGMAX_SLAB_Z) -> np.ndarray:
    """Argmax a ``(C, Z, Y, X)`` logits buffer over C, reading Z in slabs.

    ``logits`` may be a memmap-backed tensor/array; slicing ``[:, z0:z1]`` only
    makes that slab resident, so a 13 GB buffer never lands in RAM at once. The
    result is a ``uint8`` ``(Z, Y, X)`` label volume (the model has < 256
    classes).
    """
    n_classes = int(logits.shape[0])
    assert n_classes < 256, "uint8 labels require < 256 classes"
    z_total = int(logits.shape[1])
    out = np.empty(tuple(int(s) for s in logits.shape[1:]), dtype=np.uint8)
    for z0 in range(0, z_total, slab_z):
        z1 = min(z0 + slab_z, z_total)
        slab = logits[:, z0:z1]
        # torch.Tensor -> numpy view (CPU, no copy); numpy array -> as-is.
        arr = slab.numpy() if hasattr(slab, "numpy") else np.asarray(slab)
        out[z0:z1] = np.asarray(arr).argmax(0).astype(np.uint8)
    return out


def logits_to_labels_lowmem(
    predictor, logits, data_properties: dict, *, slab_z: int = DEFAULT_ARGMAX_SLAB_Z
) -> np.ndarray:
    """Low-memory equivalent of nnU-Net's logits->segmentation export.

    Mirrors ``convert_predicted_logits_to_segmentation_with_correct_shape`` but
    stays in label space (``uint8``) the whole way: argmax on the network grid
    (chunked over Z), nearest-resample the labels to the pre-resampling cropped
    shape, insert into the full uncropped volume via the stored bbox, then revert
    the preprocessing transpose. Returns the segmentation in the original input
    orientation/shape.
    """
    from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image

    plans_manager = predictor.plans_manager

    # 1. argmax on the network grid (chunked Z) -> uint8 labels.
    labels_net = chunked_argmax_z(logits, slab_z=slab_z)

    # 2. nearest-resample labels to the cropped-but-not-yet-resampled shape
    #    (the spatial grid nnU-Net would resample the probabilities to).
    target = tuple(
        int(s) for s in data_properties["shape_after_cropping_and_before_resampling"]
    )
    labels_resampled = _nearest_resample_labels(labels_net, target)

    # 3. revert cropping: drop the labels back into the full uncropped volume.
    seg = np.zeros(
        tuple(int(s) for s in data_properties["shape_before_cropping"]),
        dtype=np.uint8,
    )
    seg = insert_crop_into_image(
        seg, labels_resampled, data_properties["bbox_used_for_cropping"]
    )
    if hasattr(seg, "cpu"):  # insert_crop_into_image preserves array type
        seg = seg.cpu().numpy()
    seg = np.asarray(seg)

    # 4. revert the preprocessing transpose to the original axis ordering.
    seg = seg.transpose(plans_manager.transpose_backward)
    return np.ascontiguousarray(seg)


class MemmapLogitsPredictor:
    """Mixin overriding the sliding-window logits buffer with a disk memmap.

    Mix this in *before* :class:`nnUNetPredictor` in the MRO and set
    ``self.logits_memmap_dir`` to a writable directory before predicting. The
    accumulator ``predicted_logits`` is then a ``numpy.memmap`` shared with a
    torch tensor (in-place ``+=``/``/=`` write straight through to disk), so the
    OS — not RAM — holds the cold parts of the ~13 GB buffer. The much smaller
    ``n_predictions`` count (~0.3 GB) stays resident.
    """

    #: set by the caller before prediction; where the memmap file is created.
    logits_memmap_dir: str | None = None

    def _internal_predict_sliding_window_return_logits(
        self, data, slicers, do_on_device: bool = True
    ):
        # Copied from nnunetv2 2.8.0 nnUNetPredictor with one change: the
        # predicted_logits accumulator is memmap-backed instead of torch.zeros.
        import torch
        from queue import Queue
        from threading import Thread

        from nnunetv2.inference import predict_from_raw_data as _pr
        from nnunetv2.inference.sliding_window_prediction import compute_gaussian
        from nnunetv2.utilities.helpers import empty_cache

        if self.logits_memmap_dir is None:
            raise RuntimeError("logits_memmap_dir must be set before predicting")

        predicted_logits = n_predictions = prediction = gaussian = workon = None
        results_device = self.device if do_on_device else torch.device("cpu")

        def producer(d, slh, q):
            for s in slh:
                q.put(
                    (
                        torch.clone(
                            d[s][None], memory_format=torch.contiguous_format
                        ).to(self.device),
                        s,
                    )
                )
            q.put("end")

        try:
            empty_cache(self.device)

            data = data.to(results_device)
            queue = Queue(maxsize=2)
            t = Thread(target=producer, args=(data, slicers, queue))
            t.start()

            # --- memmap-backed accumulator (the only change vs. upstream) ---
            logits_shape = (
                self.label_manager.num_segmentation_heads,
                *data.shape[1:],
            )
            memmap_path = os.path.join(self.logits_memmap_dir, "logits.dat")
            logits_mm = np.memmap(
                memmap_path, dtype=np.float16, mode="w+", shape=logits_shape
            )
            predicted_logits = torch.from_numpy(logits_mm)
            n_predictions = torch.zeros(
                data.shape[1:], dtype=torch.half, device=results_device
            )

            if self.use_gaussian:
                gaussian = compute_gaussian(
                    tuple(self.configuration_manager.patch_size),
                    sigma_scale=1.0 / 8,
                    value_scaling_factor=10,
                    device=results_device,
                )
            else:
                gaussian = 1

            with _pr.tqdm(
                desc=None, total=len(slicers), disable=not self.allow_tqdm
            ) as pbar:
                while True:
                    item = queue.get()
                    if item == "end":
                        queue.task_done()
                        break
                    workon, sl = item
                    prediction = self._internal_maybe_mirror_and_predict(workon)[
                        0
                    ].to(results_device)

                    if self.use_gaussian:
                        prediction *= gaussian
                    predicted_logits[sl] += prediction
                    n_predictions[sl[1:]] += gaussian
                    queue.task_done()
                    pbar.update()
            queue.join()

            torch.div(predicted_logits, n_predictions, out=predicted_logits)
            if torch.any(torch.isinf(predicted_logits)):
                raise RuntimeError(
                    "Encountered inf in predicted array. Aborting... If this "
                    "problem persists, reduce value_scaling_factor in "
                    "compute_gaussian or increase the dtype of predicted_logits "
                    "to fp32"
                )
            return predicted_logits
        except Exception as e:
            del predicted_logits, n_predictions, prediction, gaussian, workon
            empty_cache(self.device)
            empty_cache(results_device)
            raise e
