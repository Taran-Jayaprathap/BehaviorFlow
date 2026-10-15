"""BehaviorFlow desktop app (runs locally in the browser; no internet needed).

    streamlit run app.py
"""
from __future__ import annotations

import io
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from bf.export import export_all
from bf.pipeline import crop, process_image, raw_table, review_queue, save_audit
from bf.rules import apply_rules, compile_rules, describe
from bf.template import Template

HERE = Path(__file__).parent
DEFAULT_TEMPLATE = HERE / "demo" / "form" / "template.json"
STATUS_TEXT = {"multiple": "More than one option marked", "faint": "Very light mark, possibly pencil", "uncertain": "Second mark nearby, low confidence"}

st.set_page_config(page_title="BehaviorFlow", page_icon="📝", layout="wide")
st.markdown(
    """<style>
    .bf-badge{display:inline-block;padding:2px 10px;border-radius:12px;font-size:0.8rem;font-weight:600}
    .bf-ok{background:#E3F1EC;color:#1F6B57}.bf-warn{background:#FFF3D6;color:#7A5300}
    .block-container{padding-top:3.2rem}
    </style>""",
    unsafe_allow_html=True,
)

ss = st.session_state
ss.setdefault("results", [])
ss.setdefault("overrides", {})
ss.setdefault("rules", {"reverse_score": [], "subscales": [], "missing_code": None})
ss.setdefault("rules_source", "")
ss.setdefault("qi", 0)


@st.cache_resource
def load_template(path: str) -> Template:
    t = Template.load(path)
    _ = t.master
    return t


with st.sidebar:
    st.markdown("## 📝 BehaviorFlow")
    st.caption("Paper instruments to analysis-ready data")
    st.markdown('<span class="bf-badge bf-ok">● 100% offline: nothing leaves this computer</span>', unsafe_allow_html=True)
    st.divider()
    with st.expander("Template file"):
        tpl_path = st.text_input("Template", str(DEFAULT_TEMPLATE), label_visibility="collapsed")
    tpl = load_template(tpl_path)
    st.write(f"**{tpl.name}**")
    st.caption(f"{len(tpl.items)} items · {sum(len(i.options) for i in tpl.items)} answer zones")
    st.image(tpl.master, caption="Blank master form", use_container_width=True)

tab_cap, tab_rev, tab_rules, tab_exp = st.tabs(["1 · Capture", "2 · Review", "3 · Scoring rules", "4 · Export"])

# ---------------- capture ----------------
with tab_cap:
    st.subheader("Add completed sheets")
    st.caption("Scans, phone photos or PDFs. No bubble sheets or corner markers needed.")
    files = st.file_uploader("Drop sheets here", type=["jpg", "jpeg", "png", "tif", "tiff", "pdf"], accept_multiple_files=True, label_visibility="collapsed")
    demo_dir = HERE / "demo" / "testset" / "scans"
    c1, c2 = st.columns([1, 3])
    run_demo = c1.button("Load demo sheets", disabled=not demo_dir.exists())
    run_up = c2.button("Process uploaded sheets", type="primary", disabled=not files)

    to_process: list[tuple[str, np.ndarray]] = []
    if run_demo:
        for p in sorted(demo_dir.glob("*.jpg"))[:12]:
            to_process.append((p.stem, cv2.imread(str(p))))
    if run_up and files:
        from bf.pipeline import load_pages

        with tempfile.TemporaryDirectory() as td:
            for f in files:
                p = Path(td) / f.name
                p.write_bytes(f.getbuffer())
                to_process += load_pages(p)
    if to_process:
        bar = st.progress(0.0, "Aligning and reading marks…")
        res = []
        for i, (sid, img) in enumerate(to_process):
            res.append(process_image(sid, img, tpl))
            bar.progress((i + 1) / len(to_process), f"Read {sid}")
        ss.results, ss.overrides, ss.qi = res, {}, 0
        bar.empty()

    if ss.results:
        res = ss.results
        n_items = sum(len(r.items) for r in res)
        queue = review_queue(res)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Sheets read", len(res))
        m2.metric("Answers read", n_items)
        m3.metric("Sent to review", f"{len(queue)} ({len(queue) / max(n_items, 1):.1%})")
        m4.metric("Seconds per sheet", f"{np.mean([r.seconds for r in res]):.2f}")
        rows = [{"Sheet": r.sheet_id, "Alignment": r.align_message, "Answers": len(r.items), "Needs review": sum(i.needs_review for i in r.items)} for r in res]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        sel = st.selectbox("Inspect a sheet", [r.sheet_id for r in res])
        r = next(x for x in res if x.sheet_id == sel)
        if r.aligned is not None:
            vis = cv2.cvtColor(r.aligned, cv2.COLOR_GRAY2BGR)
            for it, item in zip(r.items, tpl.items):
                for opt in item.options:
                    x, y, w, h = opt.box
                    if it.value == opt.value and not it.needs_review:
                        cv2.rectangle(vis, (x, y), (x + w, y + h), (87, 125, 46), 3)
                if it.needs_review:
                    x, y, w, h = it.crop_box
                    cv2.rectangle(vis, (x, y), (x + w, y + h), (61, 179, 242), 3)
            a, b = st.columns(2)
            a.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption="Aligned to master · green = read, amber = sent to review", use_container_width=True)
            b.dataframe(pd.DataFrame([{"Item": i.item_id, "Value": "" if i.value is None else str(i.value), "Status": i.status, "Confidence": i.confidence} for i in r.items]), hide_index=True, use_container_width=True, height=620)

# ---------------- review ----------------
with tab_rev:
    queue = review_queue(ss.results)
    if not queue:
        st.info("Nothing to review yet. Process some sheets first.")
    else:
        done = sum(1 for r, it in queue if (r.sheet_id, it.item_id) in ss.overrides)
        st.subheader("Review queue")
        st.progress(done / len(queue), f"{done} of {len(queue)} reviewed · only uncertain answers appear here")
        ss.qi = min(ss.qi, len(queue) - 1)
        r, it = queue[ss.qi]
        item = next(i for i in tpl.items if i.id == it.item_id)
        st.markdown(f"**{r.sheet_id} · Item {it.item_id[1:].lstrip('0')}** · {item.label}  "
                    f'<span class="bf-badge bf-warn">{STATUS_TEXT.get(it.status, it.status)}</span>', unsafe_allow_html=True)
        left, right = st.columns([3, 2])
        left.image(crop(r, it), caption="Participant's response (cropped from the aligned sheet)", use_container_width=True)
        blank_crop = tpl.master[max(0, it.crop_box[1]): it.crop_box[1] + it.crop_box[3], max(0, it.crop_box[0]): it.crop_box[0] + it.crop_box[2]]
        right.image(blank_crop, caption="Blank master for comparison", use_container_width=True)
        right.markdown(f"Predicted: **{'NULL' if it.value is None else it.value}**  ·  confidence {it.confidence:.2f}")
        cur = ss.overrides.get((r.sheet_id, it.item_id), "—")
        right.caption(f"Your decision: {cur}")
        cols = st.columns(len(item.options) + 3)
        for k, opt in enumerate(item.options):
            if cols[k].button(str(opt.value), key=f"o{k}", use_container_width=True):
                ss.overrides[(r.sheet_id, it.item_id)] = opt.value
                ss.qi = min(ss.qi + 1, len(queue) - 1)
                st.rerun()
        if cols[-3].button("Missing", use_container_width=True):
            ss.overrides[(r.sheet_id, it.item_id)] = None
            ss.qi = min(ss.qi + 1, len(queue) - 1)
            st.rerun()
        if cols[-2].button("◀ Prev", use_container_width=True):
            ss.qi = max(0, ss.qi - 1)
            st.rerun()
        if cols[-1].button("Next ▶", use_container_width=True):
            ss.qi = min(ss.qi + 1, len(queue) - 1)
            st.rerun()

# ---------------- rules ----------------
with tab_rules:
    st.subheader("Scoring rules in plain English")
    st.caption("A small language model on this computer turns your instructions into explicit rules you can check. It never sees participant data.")
    text = st.text_area("Instructions", "Reverse-score items 2, 4, 6, 9 and 11. Sum items 1-12 as wellbeing_total. Average items 2, 4, 6 and 11 as stress. True = 1, false = 0. Missing = -999.", height=110)
    if st.button("Translate rules", type="primary"):
        with st.spinner("Translating locally…"):
            ss.rules, ss.rules_source = compile_rules(text)
    if ss.rules_source:
        a, b = st.columns(2)
        a.markdown("**What BehaviorFlow will do**")
        for line in describe(ss.rules):
            a.markdown(f"- {line}")
        src = "local language model" if ss.rules_source.startswith("local-llm") else "built-in offline parser (no local model installed)"
        a.caption(f"Translated by: {src}")
        b.markdown("**Rule file (saved with the template)**")
        b.json(ss.rules)

# ---------------- export ----------------
with tab_exp:
    st.subheader("Export analysis-ready data")
    if not ss.results:
        st.info("Process some sheets first.")
    else:
        queue = review_queue(ss.results)
        pending = sum(1 for r, it in queue if (r.sheet_id, it.item_id) not in ss.overrides)
        if pending:
            st.warning(f"{pending} answers still need review; they will export as missing.")
        raw = raw_table(ss.results, tpl, ss.overrides)
        scored = apply_rules(raw, tpl, ss.rules)
        st.dataframe(scored, hide_index=True, use_container_width=True)
        if st.button("Export CSV, Excel and SPSS files + audit trail", type="primary"):
            with tempfile.TemporaryDirectory() as td:
                out = Path(td)
                export_all(scored, tpl, out)
                save_audit(ss.results, out / "audit_images")
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w") as z:
                    for p in out.rglob("*"):
                        if p.is_file():
                            z.write(p, p.relative_to(out))
            st.download_button("Download export (.zip)", buf.getvalue(), "behaviorflow_export.zip", "application/zip")
            st.success("Files: behaviorflow_data.csv, .xlsx (with codebook), .sav (SPSS) and audit_images/ with one evidence crop per answer.")
