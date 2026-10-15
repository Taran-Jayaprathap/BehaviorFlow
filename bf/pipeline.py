"""End-to-end processing: images in, raw table + review queue + audit crops out."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .align import align_to_master
from .detect import ItemResult, detect_page
from .template import Template

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".pdf"}


@dataclass
class SheetResult:
    sheet_id: str
    source: str
    aligned_ok: bool
    align_message: str
    items: list[ItemResult] = field(default_factory=list)
    aligned: np.ndarray | None = None
    seconds: float = 0.0


def load_pages(path: Path) -> list[tuple[str, np.ndarray]]:
    """Return (page_id, BGR image) for an image file or every page of a PDF."""
    if path.suffix.lower() == ".pdf":
        import pymupdf

        pages = []
        with pymupdf.open(path) as doc:
            for i, page in enumerate(doc):
                pix = page.get_pixmap(dpi=150)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR if pix.n == 3 else cv2.COLOR_RGBA2BGR)
                pages.append((f"{path.stem}_p{i + 1}", img))
        return pages
    img = cv2.imread(str(path))
    return [(path.stem, img)] if img is not None else []


def process_image(sheet_id: str, image: np.ndarray, template: Template, source: str = "") -> SheetResult:
    t0 = time.perf_counter()
    al = align_to_master(image, template.master)
    res = SheetResult(sheet_id, source, al.ok, al.message, aligned=al.aligned)
    if al.ok:
        res.items = detect_page(al.aligned, template)
    res.seconds = time.perf_counter() - t0
    return res


def process_paths(paths: list[Path], template: Template) -> list[SheetResult]:
    results = []
    for p in sorted(paths):
        for sheet_id, img in load_pages(p):
            results.append(process_image(sheet_id, img, template, str(p)))
    return results


def raw_table(results: list[SheetResult], template: Template, overrides: dict | None = None) -> pd.DataFrame:
    """One row per sheet. overrides: {(sheet_id, item_id): value} from human review."""
    overrides = overrides or {}
    rows = []
    for r in results:
        row = {"sheet_id": r.sheet_id}
        vals = {it.item_id: it for it in r.items}
        for item in template.items:
            key = (r.sheet_id, item.id)
            if key in overrides:
                row[item.id] = overrides[key]
            elif item.id in vals and not vals[item.id].needs_review:
                row[item.id] = vals[item.id].value
            else:
                row[item.id] = None
        rows.append(row)
    return pd.DataFrame(rows)


def review_queue(results: list[SheetResult]) -> list[tuple[SheetResult, ItemResult]]:
    return [(r, it) for r in results for it in r.items if it.needs_review]


def crop(result: SheetResult, item: ItemResult) -> np.ndarray:
    x, y, w, h = item.crop_box
    img = result.aligned
    x0, y0 = max(0, x), max(0, y)
    return img[y0 : y + h, x0 : x + w]


def save_audit(results: list[SheetResult], out_dir: Path) -> pd.DataFrame:
    """Write a cropped evidence image for every response, plus an index linking value to image."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in results:
        for it in r.items:
            name = f"{r.sheet_id}__{it.item_id}.png"
            cv2.imwrite(str(out_dir / name), crop(r, it))
            rows.append({"sheet_id": r.sheet_id, "item_id": it.item_id, "predicted": it.value, "status": it.status, "confidence": it.confidence, "evidence_image": name})
    idx = pd.DataFrame(rows)
    idx.to_csv(out_dir / "audit_index.csv", index=False)
    return idx
