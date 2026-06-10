"""Fixed label map for the DentalSegmentator model output.

Verified against the downloaded weights' dataset.json (Dataset112
DentalSegmentator v1.0.0): ids/names below match the model's labels exactly.
"""

from __future__ import annotations

from stomcore.mask import LabelInfo

DENTALSEGMENTATOR_LABELS: dict[int, LabelInfo] = {
    1: LabelInfo(1, "Upper Skull", (230, 200, 160)),
    2: LabelInfo(2, "Mandible", (200, 170, 130)),
    3: LabelInfo(3, "Upper Teeth", (255, 255, 240)),
    4: LabelInfo(4, "Lower Teeth", (245, 245, 230)),
    5: LabelInfo(5, "Mandibular canal", (220, 80, 80)),
}
