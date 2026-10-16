# BehaviorFlow prototype

Reads completed paper questionnaires (Likert, multiple choice, true/false) from scans or phone photos, fully offline, and exports CSV, Excel and SPSS files. Uncertain marks go to a human review queue instead of being guessed.

## Run it

```bash
pip install opencv-python-headless numpy pandas streamlit pyreadstat pymupdf openpyxl
streamlit run app.py          # opens on http://127.0.0.1:8501, local only
```

Click **Load demo sheets** on the Capture tab to try it with the bundled test sheets.

Optional local language model for the rule assistant (otherwise a built-in parser is used):

```bash
# install Ollama from ollama.com, then:
ollama pull qwen2.5:1.5b-instruct
```

## How it works

| Step | File | What it does |
|---|---|---|
| Template | `bf/template.py` | Blank master image + answer zones (JSON) |
| Align | `bf/align.py` | ORB keypoints + RANSAC homography warp any photo onto the master; no corner markers |
| Detect | `bf/detect.py` | New ink per zone minus the printed form; flags blank / multiple / faint / uncertain |
| Rules | `bf/rules.py` | Plain English → JSON rules via a local LLM (Ollama) or offline parser; applied deterministically |
| Export | `bf/export.py`, `bf/pipeline.py` | CSV, XLSX with codebook, SPSS .sav, and one evidence crop per answer |
| App | `app.py` | Capture, review queue, rules, export |

## Test results (synthetic sheets with known answers)

```bash
python3 demo/make_form.py        # demo form + template
python3 demo/simulate.py 40      # 40 tuning sheets
python3 demo/evaluate.py
```

Fresh held-out set of 100 sheets (`demo/heldout/evaluation.json`): 1,600 answers, 0 silent errors, 8.9% sent to review (12% of answers were deliberately messy), 0.49 s per sheet, 70% phone photos.

## First test on real sheets

`bf/circles.py` reads circle-the-answer forms (是/否, A–D) with no blank master or drawn zones: it finds the printed option labels itself, erases them, and checks how much of the ring around each label is enclosed by pen strokes.

Tested on 14 screenshots of a real assessment booklet marked by hand in digital ink, every answer checked by a person (`demo/real_test/`). Thresholds were tuned on items 21–60 and frozen before scoring the rest.

| Answers | Count | Read, all correct | Sent to review | Wrong |
|---|---|---|---|---|
| 是/否, items 21–60 (tuning) | 40 | 30 | 10 | 0 |
| 是/否, items 61–100 (unseen) | 40 | 28 | 12 | 0 |
| A–D, items 1–30 (unseen) | 30 | 17 | 13 | 0 |
| **Total** | **110** | **75** | **35** | **0** |

All 11 answers written as a letter instead of circled went to review. The 32% review rate is the gap to close: loose circles that stray into the next row cause most of it. The booklet screenshots are not in this repo; `results.json` has every answer.

Limits: the synthetic test marks are cleaner than real ones; paper photos of real sheets, tick-or-cross boxes and handwritten digits are not tested yet. Templates are currently written as JSON (a zone-drawing screen is on the roadmap). Free-text handwriting is out of scope.

`demo/record_demo.py` re-records the screenshots and demo video in `demo/media/`.
