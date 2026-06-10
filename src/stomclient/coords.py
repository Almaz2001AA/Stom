"""Pure mapping from widget pixel coordinates to image (col, row) pixels.

The slice image is drawn top-left anchored with KeepAspectRatio, so the
visible image occupies [0, image_w*scale] x [0, image_h*scale] where
scale = min(widget_w/image_w, widget_h/image_h).
"""

from __future__ import annotations


def widget_to_image(
    pos: tuple[float, float],
    widget_size: tuple[float, float],
    image_size: tuple[float, float],
) -> tuple[float, float] | None:
    ww, wh = widget_size
    iw, ih = image_size
    if iw <= 0 or ih <= 0:
        return None
    scale = min(ww / iw, wh / ih)
    if scale <= 0:
        return None
    col = pos[0] / scale
    row = pos[1] / scale
    if col < 0 or row < 0 or col > iw or row > ih:
        return None
    return (col, row)
