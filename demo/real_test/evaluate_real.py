"""Run the circle reader on Amber's real booklet screenshots and score it against hand-checked answers.

Usage: python3 demo/real_test/evaluate_real.py <folder with 02.jpg ... 14.jpg>
The screenshots are not in the public repo (study booklet); results are in results.json.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))
from bf.circles import draw, read_page  # noqa: E402

SRC = Path(sys.argv[1])
A = {k: cv2.imread(str(HERE / "anchors" / f"{k}.png"), 0) for k in ("yes", "no", "A", "B", "C", "D")}
YN = {"是": A["yes"], "否": A["no"]}
MC = {k: A[k] for k in "ABCD"}
# page file -> (anchor set, item ids of the answer rows top to bottom, split)
PAGES = [
    ("03.jpg", "yn", [f"yn{i}" for i in range(21, 41)], "tuning"),
    ("05.jpg", "yn", [f"yn{i}" for i in range(81, 101)], "test"),
    ("02.jpg", "yn", [f"yn{i}" for i in range(39, 61)], "tuning"),
    ("04.jpg", "yn", [f"yn{i}" for i in range(61, 81)], "test"),
    ("11.jpg", "mc", [f"mc{i}" for i in range(1, 9)], "test"),
    ("13.jpg", "mc", [f"mc{i}" for i in range(9, 16)], "test"),
    ("14.jpg", "mc", [f"mc{i}" for i in range(16, 24)], "test"),
    ("12.jpg", "mc", [f"mc{i}" for i in range(23, 31)], "test"),
]
gt = {r["item"]: r for r in csv.DictReader(open(HERE / "ground_truth.csv"))}
out, seen = [], set()
vis_dir = HERE / "overlays"
vis_dir.mkdir(exist_ok=True)
for name, kind, ids, split in PAGES:
    img = cv2.imread(str(SRC / name))
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g[:230] = 255  # phone status bar and toolbar
    g[1760:] = 255
    rows = read_page(g, YN, min_per_row=1, thr=0.55) if kind == "yn" else read_page(g, MC, min_per_row=2, thr=0.6)
    assert len(rows) == len(ids), (name, len(rows), len(ids))
    cv2.imwrite(str(vis_dir / name.replace(".jpg", ".png")), draw(img, rows))
    for item, r in zip(ids, rows):
        if item in seen:
            continue
        seen.add(item)
        t = gt[item]
        if r.status == "ok":
            outcome = "correct" if t["answer"] in (r.value, "?") else "WRONG"
        else:
            outcome = "review"
        out.append(dict(item=item, split=split, read=r.value, status=r.status, truth=t["answer"],
                        note=t["note"], outcome=outcome, coverage=r.coverage))


def summary(rows):
    n = len(rows)
    c = {k: sum(r["outcome"] == k for r in rows) for k in ("correct", "review", "WRONG")}
    return dict(answers=n, **c, review_rate=round(c["review"] / n, 3) if n else 0)


res = {
    "tuning (是/否 items 21-60)": summary([r for r in out if r["split"] == "tuning"]),
    "test (是/否 items 61-100)": summary([r for r in out if r["split"] == "test" and r["item"].startswith("yn")]),
    "test (A-D items 1-30)": summary([r for r in out if r["item"].startswith("mc")]),
    "all": summary(out),
    "handwritten letters (not circled)": summary([r for r in out if "handwritten" in r["note"]]),
    "ambiguous to a human": summary([r for r in out if r["note"].startswith("ambiguous")]),
}
json.dump(dict(summary=res, answers=out), open(HERE / "results.json", "w"), ensure_ascii=False, indent=1)
for k, v in res.items():
    print(f"{k:38s} {v}")
