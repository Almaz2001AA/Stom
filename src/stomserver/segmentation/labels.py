"""Fixed label map for the DentalSegmentator model output.

NOTE: label ids/order must match the model's dataset.json. Verify against the
downloaded weights in Task 14 and adjust names/ids if the model differs.
"""

from __future__ import annotations

from stomcore.mask import LabelInfo

DENTALSEGMENTATOR_LABELS: dict[int, LabelInfo] = {
    1: LabelInfo(1, "maxilla-upper-skull", (230, 200, 160)),
    2: LabelInfo(2, "mandible", (200, 170, 130)),
    3: LabelInfo(3, "upper-teeth", (255, 255, 240)),
    4: LabelInfo(4, "lower-teeth", (245, 245, 230)),
    5: LabelInfo(5, "mandibular-canal", (220, 80, 80)),
}
