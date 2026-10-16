"""Circle-the-answer reader for forms where people circle a printed option (是/否, A-D).

No blank master or hand-drawn zones are needed: the printed option labels are found
by template matching, erased, and each label is scored by how much of the ring
around it is enclosed by pen strokes (share of 36 rays that hit ink). Loose circles,
circles that stray into the next row, and crossings-out fall between the
thresholds and go to the review queue instead of being guessed.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .detect import _flatten

MARK = 0.45  # ring coverage that counts as circled (tuned on 40 real answers)
WEAK = 0.35  # below MARK but above this: some ink, needs a human look
_ANGLES = np.deg2rad(np.arange(0, 360, 10))


@dataclass
class CircleRow:
    y: int
    value: object
    status: str  # ok | blank | multiple | faint | uncertain
    coverage: dict
    centers: dict


def find_anchor(gray: np.ndarray, tmpl: np.ndarray, thr: float = 0.55, scales=np.arange(0.85, 1.2, 0.025)):
    """All places a printed label appears, at the best-matching scale."""
    best = None
    for s in scales:
        t = cv2.resize(tmpl, None, fx=s, fy=s)
        r = cv2.matchTemplate(gray, t, cv2.TM_CCOEFF_NORMED)
        if best is None or r.max() > best[0].max():
            best = (r, t)
    r, t = best
    h, w = t.shape
    pts = []
    while True:
        _, v, _, (x, y) = cv2.minMaxLoc(r)
        if v < thr:
            break
        pts.append((x, y))
        r[max(0, y - h):y + h, max(0, x - w):x + w] = -1
    return pts, t


def coverage(ann: np.ndarray, cx: float, cy: float, r0: int, r1: int) -> float:
    hit = 0
    for a in _ANGLES:
        ca, sa = np.cos(a), np.sin(a)
        for r in range(r0, r1):
            x, y = int(cx + r * ca), int(cy + r * sa)
            if 0 <= y < ann.shape[0] and 0 <= x < ann.shape[1] and ann[y, x]:
                hit += 1
                break
    return hit / len(_ANGLES)


def classify(cov: dict, mark: float = MARK, weak: float = WEAK):
    ranked = sorted(cov.items(), key=lambda kv: -kv[1])
    (v1, s1), (_, s2) = ranked[0], ranked[1]
    if s1 >= mark and s2 >= mark:
        return None, "multiple"
    if s1 >= mark and s2 >= weak:
        return v1, "uncertain"
    if s1 >= mark:
        return v1, "ok"
    if s1 >= weak:
        return v1, "faint"
    return None, "blank"


def read_page(gray: np.ndarray, anchors: dict, r1: int = 42, row_tol: int = 15, col_tol: int = 25,
              min_per_row: int = 1, thr: float = 0.55) -> list[CircleRow]:
    """anchors maps option value -> template image of its printed label."""
    found = {v: find_anchor(gray, t, thr) for v, t in anchors.items()}
    colx = {v: int(np.median([p[0] for p in pts])) for v, (pts, _) in found.items() if pts}
    ys = sorted(p[1] for v, (pts, _) in found.items() for p in pts if abs(p[0] - colx[v]) < col_tol)
    groups = []
    for y in ys:
        if groups and abs(y - groups[-1][-1]) < row_tol:
            groups[-1].append(y)
        else:
            groups.append([y])
    rows = [int(np.median(g)) for g in groups if len(g) >= min_per_row]

    ink = (_flatten(gray) < 200).astype(np.uint8)
    printed = np.zeros_like(ink)
    pos = {}
    for i, y in enumerate(rows):
        for v, (pts, t) in found.items():
            h, w = t.shape
            near = [p for p in pts if abs(p[1] - y) < row_tol and abs(p[0] - colx[v]) < col_tol]
            x, yy = near[0] if near else (colx[v], y)
            pos[i, v] = (x, yy, w, h)
            printed[yy:yy + h, x:x + w] |= (t < 200).astype(np.uint8)
    printed = cv2.dilate(printed, np.ones((5, 5), np.uint8))
    ann = cv2.morphologyEx(ink & (1 - printed), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    out = []
    for i, y in enumerate(rows):
        cov, ctr = {}, {}
        for v in found:
            x, yy, w, h = pos[i, v]
            ctr[v] = (x + w / 2, yy + h / 2)
            cov[v] = coverage(ann, *ctr[v], max(w, h) // 2 + 2, r1)
        value, status = classify(cov)
        out.append(CircleRow(y, value, status, {k: round(c, 2) for k, c in cov.items()}, ctr))
    return out


def draw(img: np.ndarray, rows: list[CircleRow], radius: int = 22) -> np.ndarray:
    vis = img.copy()
    for row in rows:
        for v, (cx, cy) in row.centers.items():
            if row.status == "ok" and v == row.value:
                c = (0, 160, 0)
            elif row.status != "ok" and row.coverage[v] >= WEAK:
                c = (0, 0, 220)
            else:
                c = (190, 190, 190)
            cv2.circle(vis, (int(cx), int(cy)), radius, c, 3 if c != (190, 190, 190) else 1)
        x = int(max(cx for cx, _ in row.centers.values())) + 30
        label = "read" if row.status == "ok" else "review"  # cv2 cannot draw CJK option labels
        cv2.putText(vis, label, (x, row.y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 140, 0) if row.status == "ok" else (0, 0, 220), 1)
    return vis
