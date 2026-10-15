"""Markerless alignment: warp a scan or phone photo onto the master template.

Uses ORB keypoints on the printed text of the form itself, so no corner
markers or bubble grids are needed.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class AlignResult:
    aligned: np.ndarray | None  # grayscale, same size as master
    inliers: int
    inlier_ratio: float
    ok: bool
    message: str


def _prep(gray: np.ndarray) -> np.ndarray:
    # Contrast-limited equalisation makes keypoints stable under shadows.
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def align_to_master(image: np.ndarray, master: np.ndarray, min_inliers: int = 60) -> AlignResult:
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Work at master scale so feature sizes are comparable.
    scale = master.shape[1] * 1.6 / max(gray.shape[1], 1)
    if scale < 1.0:
        work = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        scale, work = 1.0, gray

    orb = cv2.ORB_create(nfeatures=8000, scaleFactor=1.2, nlevels=8)
    k1, d1 = orb.detectAndCompute(_prep(work), None)
    k2, d2 = orb.detectAndCompute(_prep(master), None)
    if d1 is None or d2 is None or len(k1) < 20:
        return AlignResult(None, 0, 0.0, False, "Too few features: image may be blank or very blurry")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    good = []
    for pair in matcher.knnMatch(d1, d2, k=2):
        if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
            good.append(pair[0])
    if len(good) < min_inliers:
        return AlignResult(None, len(good), 0.0, False, "Not enough matches to the master: is this the right form?")

    src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2) / scale
    dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if H is None:
        return AlignResult(None, 0, 0.0, False, "Could not compute page geometry")

    inliers = int(mask.sum())
    ratio = inliers / len(good)
    h, w = master.shape[:2]
    aligned = cv2.warpPerspective(gray, H, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)
    ok = inliers >= min_inliers
    msg = "Aligned" if ok else "Low alignment quality: please rescan"
    return AlignResult(aligned, inliers, ratio, ok, msg)
