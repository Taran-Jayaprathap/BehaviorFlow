"""Create the demo instrument: a blank master page and its template JSON.

The form is a generic 16-item questionnaire written for this demo (12 Likert
items, 4 true/false items), in an ordinary printed layout with no bubbles or
corner markers.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bf.template import Item, Option, Template  # noqa: E402

W, H = 1240, 1754  # A4 at 150 dpi
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
SERIF_B = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"

LIKERT = [
    "I felt calm and relaxed this week.",
    "I found it hard to concentrate on my studies.",
    "I had enough energy for daily tasks.",
    "I worried about things I could not control.",
    "I felt connected to people around me.",
    "Small problems made me feel overwhelmed.",
    "I slept well most nights.",
    "I felt confident handling unexpected events.",
    "I avoided tasks because they felt too difficult.",
    "I enjoyed activities I usually like.",
    "I felt tense or on edge.",
    "I was satisfied with how I spent my time.",
]
TF = [
    "I attended at least one social event this week.",
    "I skipped a meal because I was too busy.",
    "I exercised for 30 minutes or more on most days.",
    "I used a phone or computer within an hour of bedtime.",
]

OPT_X0, OPT_DX = 770, 92


def build(out_dir: Path) -> Template:
    out_dir.mkdir(parents=True, exist_ok=True)
    img = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(img)
    f_title, f_b, f, f_s, f_num = (ImageFont.truetype(SERIF_B, 40), ImageFont.truetype(FONT_B, 24), ImageFont.truetype(FONT, 22), ImageFont.truetype(FONT, 17), ImageFont.truetype(FONT, 30))

    d.text((90, 80), "Weekly Wellbeing Questionnaire", font=f_title, fill=0)
    d.text((90, 140), "BehaviorFlow demo instrument · Form WWQ-16 · Version 1.2", font=f_s, fill=60)
    d.text((90, 185), "Participant ID: ______________      Session: ____      Date: ____________", font=f, fill=0)
    d.text((90, 240), "Part A. Circle ONE number for each statement about the past 7 days.", font=f_b, fill=0)
    d.text((OPT_X0 - 52, 285), "Strongly\ndisagree", font=f_s, fill=40, align="center")
    d.text((OPT_X0 + 4 * OPT_DX - 34, 285), "Strongly\nagree", font=f_s, fill=40, align="center")

    items = []
    y = 360
    for i, text in enumerate(LIKERT):
        d.text((90, y + 6), f"{i + 1}.", font=f, fill=0)
        d.text((135, y + 6), text, font=f, fill=0)
        opts = []
        for k in range(5):
            cx, cy = OPT_X0 + k * OPT_DX, y + 20
            d.text((cx, cy), str(k + 1), font=f_num, fill=0, anchor="mm")
            opts.append(Option(k + 1, (cx - 35, cy - 30, 70, 60)))
        items.append(Item(f"q{i + 1:02d}", text, "likert", opts))
        if i < len(LIKERT) - 1:
            d.line((90, y + 58, W - 90, y + 58), fill=215, width=1)
        y += 76

    y += 40
    d.text((90, y), "Part B. Circle T (true) or F (false) for each statement.", font=f_b, fill=0)
    y += 60
    for j, text in enumerate(TF):
        n = len(LIKERT) + j + 1
        d.text((90, y + 6), f"{n}.", font=f, fill=0)
        d.text((135, y + 6), text, font=f, fill=0)
        opts = []
        for k, v in enumerate(["T", "F"]):
            cx, cy = OPT_X0 + k * OPT_DX, y + 20
            d.text((cx, cy), v, font=f_num, fill=0, anchor="mm")
            opts.append(Option(v, (cx - 35, cy - 30, 70, 60)))
        items.append(Item(f"q{n:02d}", text, "true_false", opts))
        y += 76

    d.text((90, H - 90), "Thank you for taking part. Please hand this sheet to the researcher.", font=f_s, fill=60)
    d.text((W - 200, H - 90), "Page 1 of 1", font=f_s, fill=60)

    master = out_dir / "master.png"
    img.save(master)
    img.convert("RGB").save(out_dir / "master_blank_form.pdf", resolution=150)
    tpl = Template("Weekly Wellbeing Questionnaire (WWQ-16)", master, items)
    tpl.save(out_dir / "template.json")
    return tpl


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "form"
    t = build(out)
    print(f"Wrote {out}/master.png and template.json with {len(t.items)} items")
