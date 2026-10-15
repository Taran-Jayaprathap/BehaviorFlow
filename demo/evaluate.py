"""Run BehaviorFlow on the test sheets and compare with the known answers.

Reports:
  - auto-accepted accuracy: of the answers the tool accepted on its own, how many were right
  - review rate: share of answers sent to a human
  - silent errors: wrong answers that were NOT flagged (the number that matters for data quality)
  - seconds per sheet on this machine
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bf.pipeline import process_paths  # noqa: E402
from bf.template import Template  # noqa: E402


def norm(v):
    if v is None or (isinstance(v, float) and pd.isna(v)) or v == "":
        return ""
    return str(v)


def main(template_path: Path, testset: Path):
    tpl = Template.load(template_path)
    truth = pd.read_csv(testset / "ground_truth.csv", dtype=str).fillna("")
    results = process_paths(list((testset / "scans").glob("*.jpg")), tpl)

    rows = []
    for r in results:
        for it in r.items:
            rows.append({"sheet_id": r.sheet_id, "item_id": it.item_id, "pred": norm(it.value), "status": it.status, "review": it.needs_review, "conf": it.confidence})
    pred = pd.DataFrame(rows)
    df = truth.merge(pred, on=["sheet_id", "item_id"], how="left")
    failed = [r.sheet_id for r in results if not r.aligned_ok]

    df["correct"] = df["pred"] == df["truth"]
    auto = df[~df["review"].astype(bool)]
    silent = auto[~auto["correct"]]
    summary = {
        "sheets": len(results),
        "sheets_failed_alignment": failed,
        "answers": len(df),
        "review_rate": round(df["review"].mean(), 4),
        "auto_accepted": len(auto),
        "auto_accuracy": round(auto["correct"].mean(), 4),
        "silent_errors": len(silent),
        "accuracy_after_review": round(1 - len(silent) / len(df), 4),
        "mean_seconds_per_sheet": round(sum(r.seconds for r in results) / len(results), 2),
        "messy_answers": int((df["kind"] != "normal").sum()),
        "messy_flagged": int(df[(df["kind"].isin(["double", "faint", "straddle"]))]["review"].sum()),
        "messy_needing_review": int(df["kind"].isin(["double", "faint", "straddle"]).sum()),
        "by_capture": df.groupby("capture").apply(lambda g: {"answers": len(g), "review_rate": round(g["review"].mean(), 4), "silent_errors": int((~g["correct"] & ~g["review"].astype(bool)).sum())}).to_dict(),
    }
    (testset / "evaluation.json").write_text(json.dumps(summary, indent=2, default=str))
    df.to_csv(testset / "evaluation_detail.csv", index=False)
    print(json.dumps(summary, indent=2, default=str))
    if len(silent):
        print(silent[["sheet_id", "item_id", "truth", "pred", "status", "kind", "capture"]].to_string())


if __name__ == "__main__":
    here = Path(__file__).parent
    main(here / "form" / "template.json", here / "testset")
