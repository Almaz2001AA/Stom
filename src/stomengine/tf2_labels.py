"""Label map for the ToothFairy2 (Dataset112_ToothFairy2) 49-class model.

Verified against the pretrained checkpoint's ``dataset.json``: ids 1-9 are
anatomy, 10 is implants, and 11-48 are individual teeth in **FDI** numbering
(11-18 upper-right, 21-28 upper-left, 31-38 lower-left, 41-48 lower-right). The
gaps 19/20, 29/30, 39/40 are unused placeholders the dataset keeps so the tooth
ids line up with FDI exactly; the model never emits them, so they are omitted.

Unlike DentalSegmentator (5 coarse classes, teeth as one block), this model
segments every tooth separately — the path to per-tooth findings/STL.
"""

from __future__ import annotations

from stomcore.mask import LabelInfo

# Anatomy / non-tooth structures (ids 1-10).
_ANATOMY: dict[int, tuple[str, tuple[int, int, int]]] = {
    1: ("Lower Jawbone", (200, 170, 130)),
    2: ("Upper Jawbone", (230, 200, 160)),
    3: ("Left Inferior Alveolar Canal", (220, 80, 80)),
    4: ("Right Inferior Alveolar Canal", (220, 120, 80)),
    5: ("Left Maxillary Sinus", (120, 180, 230)),
    6: ("Right Maxillary Sinus", (120, 210, 200)),
    7: ("Pharynx", (180, 130, 200)),
    8: ("Bridge", (240, 220, 120)),
    9: ("Crown", (250, 235, 180)),
    10: ("Implant", (160, 160, 170)),
}

# Per-tooth FDI ids -> human name. Quadrants: 1x upper-right, 2x upper-left,
# 3x lower-left, 4x lower-right; positions 1-8 = central incisor .. third molar.
_POSITION = {
    1: "Central Incisor",
    2: "Lateral Incisor",
    3: "Canine",
    4: "First Premolar",
    5: "Second Premolar",
    6: "First Molar",
    7: "Second Molar",
    8: "Third Molar (Wisdom Tooth)",
}
_QUADRANT = {1: "Upper Right", 2: "Upper Left", 3: "Lower Left", 4: "Lower Right"}


def _tooth_color(fdi: int) -> tuple[int, int, int]:
    """A distinct ivory-ish shade per tooth so adjacent teeth are separable."""
    quadrant = fdi // 10
    position = fdi % 10
    # Warm ivory base nudged by quadrant (hue) and position (lightness).
    r = 255 - (position - 1) * 4
    g = 245 - (quadrant - 1) * 10 - (position - 1) * 3
    b = 200 + (quadrant - 1) * 12 + (position - 1) * 5
    clamp = lambda v: max(0, min(255, v))  # noqa: E731
    return (clamp(r), clamp(g), clamp(b))


def _build() -> dict[int, LabelInfo]:
    labels: dict[int, LabelInfo] = {
        i: LabelInfo(i, name, color) for i, (name, color) in _ANATOMY.items()
    }
    for quadrant, q_name in _QUADRANT.items():
        for position, p_name in _POSITION.items():
            fdi = quadrant * 10 + position
            labels[fdi] = LabelInfo(
                fdi, f"{q_name} {p_name}", _tooth_color(fdi)
            )
    return labels


TOOTHFAIRY2_LABELS: dict[int, LabelInfo] = _build()
