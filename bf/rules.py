"""Scoring rules: plain English in, explicit JSON rules out.

The researcher types rules such as
    "Reverse-score items 3, 7 and 12. Sum items 1-6 as anxiety. Missing = -999."
A small language model running locally (through Ollama on localhost, so no
data leaves the machine) translates them into the JSON below. The model only
writes the rule set; it never sees participant data, and the rules are
applied by plain deterministic code, so results are reproducible.

If no local model is installed, a pattern-based parser handles the common
phrasings so the tool still works fully offline.

Rule JSON:
{
  "reverse_score": [{"items": [3, 7, 12]}],          # 1-based item numbers
  "subscales": [{"name": "anxiety", "items": [1,2,3], "method": "sum"}],
  "missing_code": -999,
  "true_false": {"T": 1, "F": 0}
}
"""
from __future__ import annotations

import json
import os
import re
import urllib.request

import pandas as pd

from .template import Template

OLLAMA_URL = os.environ.get("BF_OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
LLM_MODEL = os.environ.get("BF_LLM_MODEL", "qwen2.5:1.5b-instruct")

RULE_SCHEMA = {
    "type": "object",
    "properties": {
        "reverse_score": {"type": "array", "items": {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "integer"}}}, "required": ["items"]}},
        "subscales": {"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}, "items": {"type": "array", "items": {"type": "integer"}}, "method": {"type": "string", "enum": ["sum", "mean"]}}, "required": ["name", "items", "method"]}},
        "missing_code": {"type": ["integer", "null"]},
        "true_false": {"type": "object", "properties": {"T": {"type": "integer"}, "F": {"type": "integer"}}},
    },
    "required": ["reverse_score", "subscales", "missing_code"],
}

PROMPT = """You convert a researcher's scoring instructions for a paper questionnaire into JSON.
Item numbers are 1-based positions on the form. Only use what the instructions say.
Return JSON with keys: reverse_score (list of {{"items": [ints]}}), subscales (list of
{{"name", "items": [ints], "method": "sum" or "mean"}}), missing_code (int or null),
true_false ({{"T": int, "F": int}}, optional).

Instructions:
{text}
"""


def _numbers(spec: str) -> list[int]:
    spec = spec.replace("–", "-").replace(" to ", "-")
    out: list[int] = []
    for a, b in re.findall(r"(\d+)\s*(?:-\s*(\d+))?", spec):
        if b:
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(a))
    return out


NUMLIST = r"((?:\d+(?:\s*(?:-|–|to)\s*\d+)?(?:\s*,\s*|\s*,?\s*and\s*|\s+))*\d+(?:\s*(?:-|–|to)\s*\d+)?)"


def parse_rules_offline(text: str) -> dict:
    t = text.lower()
    rules = {"reverse_score": [], "subscales": [], "missing_code": None}
    for m in re.finditer(r"reverse[- ]?(?:score|scored|code|coded|scoring)?\s*(?:items?|questions?|q)?\s*" + NUMLIST, t):
        rules["reverse_score"].append({"items": _numbers(m.group(1))})
    for m in re.finditer(r"(sum|total|add up|average|mean)\s*(?:of\s*)?(?:items?|questions?|q)\s*" + NUMLIST + r"\s*(?:as|into|for|called|=)\s*[\"']?([a-z][a-z _]*)", t):
        method = "mean" if m.group(1) in ("average", "mean") else "sum"
        rules["subscales"].append({"name": m.group(3).strip().replace(" ", "_"), "items": _numbers(m.group(2)), "method": method})
    m = re.search(r"missing[^0-9\-]{0,25}(-?\d+)", t)
    if m:
        rules["missing_code"] = int(m.group(1))
    m_t, m_f = re.search(r"true\s*(?:=|as|is)\s*(-?\d+)", t), re.search(r"false\s*(?:=|as|is)\s*(-?\d+)", t)
    if m_t and m_f:
        rules["true_false"] = {"T": int(m_t.group(1)), "F": int(m_f.group(1))}
    return rules


def _valid(rules: dict) -> bool:
    try:
        assert isinstance(rules.get("reverse_score"), list) and isinstance(rules.get("subscales"), list)
        for r in rules["reverse_score"]:
            assert all(isinstance(i, int) for i in r["items"])
        for s in rules["subscales"]:
            assert s["method"] in ("sum", "mean") and all(isinstance(i, int) for i in s["items"])
        return True
    except (AssertionError, KeyError, TypeError):
        return False


def compile_rules(text: str, timeout: float = 60) -> tuple[dict, str]:
    """Return (rules, source) where source is 'local-llm:<model>' or 'offline-parser'."""
    body = json.dumps({"model": LLM_MODEL, "prompt": PROMPT.format(text=text), "format": RULE_SCHEMA, "stream": False, "options": {"temperature": 0}}).encode()
    try:
        req = urllib.request.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            rules = json.loads(json.loads(resp.read())["response"])
        if _valid(rules):
            return rules, f"local-llm:{LLM_MODEL}"
    except Exception:
        pass
    return parse_rules_offline(text), "offline-parser"


def describe(rules: dict) -> list[str]:
    lines = []
    for r in rules.get("reverse_score", []):
        lines.append(f"Reverse-score items {', '.join(map(str, r['items']))} (new value = min + max - old)")
    for s in rules.get("subscales", []):
        lines.append(f"{s['name']} = {s['method']} of items {', '.join(map(str, s['items']))}")
    if rules.get("missing_code") is not None:
        lines.append(f"Missing responses coded as {rules['missing_code']}")
    if rules.get("true_false"):
        lines.append(f"True = {rules['true_false']['T']}, False = {rules['true_false']['F']}")
    return lines or ["No rules: raw values exported as read"]


def apply_rules(raw: pd.DataFrame, template: Template, rules: dict) -> pd.DataFrame:
    """raw: one row per sheet, one column per item id (values or NaN). Returns the scored table."""
    df = raw.copy()
    ids = [it.id for it in template.items]
    by_num = {i + 1: it for i, it in enumerate(template.items)}

    tf = rules.get("true_false")
    if tf:
        for it in template.items:
            if it.type == "true_false":
                df[it.id] = df[it.id].map(lambda v: tf.get(v, v) if isinstance(v, str) else v)

    for r in rules.get("reverse_score", []):
        for n in r["items"]:
            it = by_num.get(n)
            if it is None:
                continue
            vals = [o.value for o in it.options if isinstance(o.value, (int, float))]
            lo, hi = min(vals), max(vals)
            df[f"{it.id}_r"] = pd.to_numeric(df[it.id], errors="coerce").map(lambda v: lo + hi - v if pd.notna(v) else v)

    for s in rules.get("subscales", []):
        cols = []
        rev = {n for r in rules.get("reverse_score", []) for n in r["items"]}
        for n in s["items"]:
            it = by_num.get(n)
            if it is not None:
                cols.append(f"{it.id}_r" if n in rev and f"{it.id}_r" in df else it.id)
        block = df[cols].apply(pd.to_numeric, errors="coerce")
        df[s["name"]] = block.sum(axis=1, min_count=len(cols)) if s["method"] == "sum" else block.mean(axis=1)

    if rules.get("missing_code") is not None:
        code = rules["missing_code"]
        item_cols = [c for c in df.columns if c in ids or c.endswith("_r")]
        df[item_cols] = df[item_cols].astype(object).where(df[item_cols].notna(), code)
    return df
