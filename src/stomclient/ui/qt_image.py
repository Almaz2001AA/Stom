"""Convert numpy RGB arrays to QImage."""

from __future__ import annotations

import numpy as np
from PySide6.QtGui import QImage


def ndarray_to_qimage(rgb: np.ndarray) -> QImage:
    """rgb: [row, col, 3] uint8, C-contiguous. Returns an owned QImage copy."""
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w, _ = rgb.shape
    img = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return img.copy()  # detach from the numpy buffer
