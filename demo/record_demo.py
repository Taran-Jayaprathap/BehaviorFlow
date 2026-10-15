"""Drive the running app (streamlit run app.py) and save screenshots plus a screen recording."""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:8501"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "media"
OUT.mkdir(parents=True, exist_ok=True)
EXE = "/opt/pw-browsers/chromium"


def settle(page, t=1.5):
    page.wait_for_timeout(300)
    try:
        page.wait_for_selector('[data-testid="stStatusWidget"]', state="detached", timeout=60000)
    except Exception:
        pass
    page.wait_for_timeout(int(t * 1500))


with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=EXE)
    ctx = browser.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=1.5)
    page = ctx.new_page()
    # Screen recording via the DevTools screencast: frames are stamped and later stitched with ffmpeg.
    frames = []
    cdp = ctx.new_cdp_session(page)

    def on_frame(ev):
        frames.append((ev["metadata"]["timestamp"], ev["data"]))
        cdp.send("Page.screencastFrameAck", {"sessionId": ev["sessionId"]})

    cdp.on("Page.screencastFrame", on_frame)
    cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 85, "maxWidth": 1920, "maxHeight": 1200, "everyNthFrame": 1})
    page.goto(URL)
    page.get_by_text("Load demo sheets").wait_for(timeout=60000)
    settle(page, 2)
    page.screenshot(path=str(OUT / "0_start.png"))

    page.get_by_role("button", name="Load demo sheets").click()
    page.get_by_text("Sheets read").wait_for(timeout=120000)
    settle(page, 2)
    page.screenshot(path=str(OUT / "1_capture.png"))
    # show a sheet that has answers flagged for review
    page.locator('[data-testid="stSelectbox"]').click()
    page.get_by_role("option", name="P1008").click()
    settle(page, 1)
    page.mouse.wheel(0, 700)
    settle(page, 2)
    page.screenshot(path=str(OUT / "2_inspect.png"))
    page.mouse.wheel(0, -2000)

    page.get_by_role("tab", name="2 · Review").click()
    settle(page, 2)
    page.screenshot(path=str(OUT / "3_review.png"))
    for _ in range(3):
        page.get_by_role("button", name="Next ▶").wait_for()
        # answer with the first option shown for this item
        page.get_by_role("button", name=re.compile(r"^(1|T)$")).first.click()
        settle(page, 1.2)
    page.screenshot(path=str(OUT / "4_review_progress.png"))

    page.get_by_role("tab", name="3 · Scoring rules").click()
    settle(page, 1)
    page.get_by_role("button", name="Translate rules").click()
    page.get_by_text("What BehaviorFlow will do").wait_for(timeout=120000)
    settle(page, 2)
    page.screenshot(path=str(OUT / "5_rules.png"))

    page.get_by_role("tab", name="4 · Export").click()
    settle(page, 2)
    page.get_by_role("button", name="Export CSV, Excel and SPSS files + audit trail").click()
    page.get_by_text("Download export (.zip)").wait_for(timeout=120000)
    settle(page, 2)
    page.screenshot(path=str(OUT / "6_export.png"))
    page.wait_for_timeout(2500)
    cdp.send("Page.stopScreencast")
    import base64
    fdir = OUT / "frames"
    fdir.mkdir(exist_ok=True)
    lines = []
    for i, (ts, data) in enumerate(frames):
        (fdir / f"f{i:05d}.jpg").write_bytes(base64.b64decode(data))
        dur = (frames[i + 1][0] - ts) if i + 1 < len(frames) else 2.0
        lines.append(f"file 'f{i:05d}.jpg'\nduration {max(dur, 0.04):.3f}")
    lines.append(f"file 'f{len(frames) - 1:05d}.jpg'")
    (fdir / "list.txt").write_text("\n".join(lines))
    print("frames", len(frames))
    ctx.close()
    browser.close()
print("saved to", OUT)
