"""Make realistic filled test sheets with known answers.

Each sheet is the master form with hand-drawn-style marks (circles, ticks,
crosses in pen or pencil), then degraded like a real capture: phone photos
with perspective, rotation, shadows, blur, noise and JPEG compression, or
slightly skewed flatbed scans. Some responses are deliberately messy:
blank, double-marked, faint pencil, or circled between two options.

Writes scans/*.jpg and ground_truth.csv (sheet_id, item_id, truth, kind).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bf.template import Template  # noqa: E402

PENS = {"blue": (150, 60, 20), "black": (35, 35, 35), "pencil": (105, 105, 105)}


def _circle(img, c, rng, color, th):
    rx, ry = rng.uniform(26, 34), rng.uniform(22, 29)
    a0 = rng.uniform(0, 2 * np.pi)
    sweep = rng.uniform(2.0 * np.pi * 0.95, 2.0 * np.pi * 1.15)
    t = np.linspace(a0, a0 + sweep, 80)
    wob = 1 + 0.06 * np.sin(t * rng.uniform(2, 4) + rng.uniform(0, 6))
    pts = np.stack([c[0] + rx * wob * np.cos(t), c[1] + ry * wob * np.sin(t)], 1)
    cv2.polylines(img, [pts.astype(np.int32)], False, color, th, cv2.LINE_AA)


def _tick(img, c, rng, color, th):
    x, y = c
    p = np.array([[x - 18, y], [x - 5, y + 16], [x + 22, y - 20]], np.float32) + rng.normal(0, 2, (3, 2))
    cv2.polylines(img, [p.astype(np.int32)], False, color, th, cv2.LINE_AA)


def _cross(img, c, rng, color, th):
    x, y = c
    for a, b in (((-18, -18), (18, 18)), ((18, -18), (-18, 18))):
        p1 = (int(x + a[0] + rng.normal(0, 2)), int(y + a[1] + rng.normal(0, 2)))
        p2 = (int(x + b[0] + rng.normal(0, 2)), int(y + b[1] + rng.normal(0, 2)))
        cv2.line(img, p1, p2, color, th, cv2.LINE_AA)


def _center(opt, rng):
    x, y, w, h = opt.box
    return (x + w / 2 + rng.normal(0, 3), y + h / 2 + rng.normal(0, 3))


def fill_sheet(tpl: Template, rng: np.random.Generator, messy_rate: float):
    page = cv2.cvtColor(tpl.master, cv2.COLOR_GRAY2BGR)
    pen = rng.choice(list(PENS))
    color = PENS[pen]
    th = int(rng.integers(2, 4))
    style = rng.choice(["circle", "circle", "circle", "tick", "cross"])
    draw = {"circle": _circle, "tick": _tick, "cross": _cross}[style]
    truth = []
    for it in tpl.items:
        opts = it.options
        k = int(rng.integers(0, len(opts)))
        kind = "normal"
        r = rng.random()
        if r < messy_rate:
            kind = rng.choice(["blank", "double", "faint", "straddle"], p=[0.35, 0.25, 0.25, 0.15])
        if kind == "straddle" and k == len(opts) - 1:
            k -= 1
        if kind == "blank":
            truth.append((it.id, "", kind))
            continue
        if kind == "double":
            k2 = (k + int(rng.integers(1, len(opts)))) % len(opts)
            draw(page, _center(opts[k], rng), rng, color, th)
            draw(page, _center(opts[k2], rng), rng, color, th)
            truth.append((it.id, "", kind))
            continue
        if kind == "faint":
            _circle(page, _center(opts[k], rng), rng, (175, 175, 175), 1)
            truth.append((it.id, opts[k].value, kind))
            continue
        if kind == "straddle":
            a, b = _center(opts[k], rng), _center(opts[k + 1], rng)
            _circle(page, ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2), rng, color, th)
            truth.append((it.id, "", kind))
            continue
        draw(page, _center(opts[k], rng), rng, color, th)
        truth.append((it.id, opts[k].value, kind))
    return page, truth, f"{pen}/{style}"


def photograph(page: np.ndarray, rng: np.random.Generator, mode: str) -> np.ndarray:
    h, w = page.shape[:2]
    if mode == "phone":
        CW, CH = 1500, 2000
        bg = np.full((CH, CW, 3), rng.integers(60, 120), np.uint8)
        bg = cv2.add(bg, rng.normal(0, 8, bg.shape).astype(np.int16).clip(0, 255).astype(np.uint8))
        s = rng.uniform(0.95, 1.08)
        bx, by = (CW - w * s) / 2, (CH - h * s) / 2
        base = np.float32([[bx, by], [bx + w * s, by], [bx + w * s, by + h * s], [bx, by + h * s]])
        ang = np.deg2rad(rng.uniform(-8, 8))
        R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        ctr = base.mean(0)
        dst = ((base - ctr) @ R.T + ctr + rng.normal(0, 25, (4, 2))).astype(np.float32)
        M = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), dst)
        warped = cv2.warpPerspective(page, M, (CW, CH), borderValue=(0, 0, 0))
        mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, (CW, CH))
        out = np.where(mask[..., None] > 0, warped, bg).astype(np.float32)
        # uneven light / shadow
        gx = np.linspace(rng.uniform(0.55, 0.85), rng.uniform(0.9, 1.05), CW)
        gy = np.linspace(rng.uniform(0.7, 1.0), rng.uniform(0.85, 1.05), CH)
        out *= np.outer(gy, gx)[..., None]
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.6, 1.3))
        out += rng.normal(0, 5, out.shape)
        img = out.clip(0, 255).astype(np.uint8)
        q = int(rng.integers(55, 85))
    else:  # flatbed scan, slight skew
        ang = rng.uniform(-2.5, 2.5)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        M[:, 2] += rng.normal(0, 10, 2)
        img = cv2.warpAffine(page, M, (w, h), borderValue=(250, 250, 250))
        img = cv2.GaussianBlur(img, (0, 0), 0.5)
        q = 90
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def main(template_path: Path, out_dir: Path, n: int = 40, seed: int = 7, messy_rate: float = 0.12):
    tpl = Template.load(template_path)
    rng = np.random.default_rng(seed)
    scans = out_dir / "scans"
    scans.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        sid = f"P{1001 + i}"
        page, truth, desc = fill_sheet(tpl, rng, messy_rate)
        mode = "phone" if rng.random() < 0.7 else "scan"
        cv2.imwrite(str(scans / f"{sid}.jpg"), photograph(page, rng, mode))
        rows += [{"sheet_id": sid, "item_id": iid, "truth": v, "kind": k, "capture": mode, "marker": desc} for iid, v, k in truth]
    with open(out_dir / "ground_truth.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"Wrote {n} sheets to {scans}")


if __name__ == "__main__":
    here = Path(__file__).parent
    main(here / "form" / "template.json", here / "testset", n=int(sys.argv[1]) if len(sys.argv) > 1 else 40)
