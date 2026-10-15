"""Mark detection: find which option in each item the participant marked.

Works on the aligned page. Ink that is already printed on the blank master
(digits, labels) is subtracted, so only the participant's pen or pencil marks
count. Each item gets a value, a status and a confidence; anything uncertain
is routed to human review instead of being guessed.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .template import Item, Template

# Share of pixels in a zone that must be new ink to count as a mark.
MARK_T = 0.035
# Below this a zone is clean; between FAINT_T and MARK_T it is ambiguous.
FAINT_T = 0.012
# Light-pencil score above which an apparently blank item is sent to review.
LIGHT_T = 0.02


@dataclass
class ItemResult:
    item_id: str
    value: object  # predicted value, or None
    status: str  # ok | blank | multiple | faint | uncertain
    confidence: float  # 0..1
    scores: dict  # option value -> ink score
    crop_box: tuple[int, int, int, int]  # region covering all options

    @property
    def needs_review(self) -> bool:
        return self.status in ("multiple", "faint", "uncertain")


def _flatten(gray: np.ndarray) -> np.ndarray:
    """Remove uneven lighting (phone shadows) by dividing by a background estimate."""
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    flat = cv2.divide(gray, bg, scale=255)
    return flat


def ink_mask(gray: np.ndarray, level: int = 200) -> np.ndarray:
    flat = _flatten(gray)
    return (flat < level).astype(np.uint8)


def new_ink(aligned: np.ndarray, master: np.ndarray, level: int = 200) -> np.ndarray:
    """Pixels that are ink on the filled page but not on the blank master.

    level=200 finds pen marks; level=230 also finds light pencil, used only to
    catch faint marks that would otherwise be read as blank.
    """
    printed = cv2.dilate(ink_mask(master, 235), np.ones((5, 5), np.uint8), iterations=1)
    filled = ink_mask(aligned, level)
    diff = filled & (1 - printed)
    # drop salt noise from scanning
    return cv2.morphologyEx(diff, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def _zone(box, pad, shape):
    x, y, w, h = box
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(shape[1], x + w + pad), min(shape[0], y + h + pad)
    return x0, y0, x1, y1


def score_item(diff: np.ndarray, item: Item, pad: int = 4) -> dict:
    scores = {}
    for opt in item.options:
        x0, y0, x1, y1 = _zone(opt.box, pad, diff.shape)
        region = diff[y0:y1, x0:x1]
        scores[opt.value] = float(region.mean()) if region.size else 0.0
    return scores


def classify(scores: dict, light: dict | None = None) -> tuple[object, str, float]:
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    (v1, s1), (_, s2) = ranked[0], ranked[1] if len(ranked) > 1 else (None, 0.0)
    marked = [v for v, s in ranked if s >= MARK_T]

    if not marked:
        lv, ls = max((light or scores).items(), key=lambda kv: kv[1])
        if s1 >= FAINT_T or ls >= LIGHT_T:
            return lv, "faint", round(min(0.49, ls / MARK_T * 0.25), 3)
        return None, "blank", round(1 - max(s1 / FAINT_T, ls / LIGHT_T) * 0.5, 3)
    if len(marked) > 1:
        return None, "multiple", round(min(1.0, s2 / s1), 3)
    margin = (s1 - s2) / s1
    if s2 >= FAINT_T and margin < 0.6:
        return v1, "uncertain", round(margin, 3)
    return v1, "ok", round(min(1.0, 0.5 + margin / 2), 3)


def detect_page(aligned: np.ndarray, template: Template) -> list[ItemResult]:
    diff = new_ink(aligned, template.master)
    light = new_ink(aligned, template.master, level=230)
    out = []
    for item in template.items:
        scores = score_item(diff, item)
        value, status, conf = classify(scores, score_item(light, item))
        xs = [o.box[0] for o in item.options] + [o.box[0] + o.box[2] for o in item.options]
        ys = [o.box[1] for o in item.options] + [o.box[1] + o.box[3] for o in item.options]
        crop = (min(xs) - 30, min(ys) - 25, max(xs) - min(xs) + 60, max(ys) - min(ys) + 50)
        out.append(ItemResult(item.id, value, status, conf, scores, crop))
    return out
