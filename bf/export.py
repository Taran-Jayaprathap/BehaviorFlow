"""Export to the formats labs analyse in: CSV, Excel, SPSS (.sav) and R-friendly CSV."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .template import Template


def export_all(scored: pd.DataFrame, template: Template, out_dir: Path, stem: str = "behaviorflow_data") -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    csv = out_dir / f"{stem}.csv"
    scored.to_csv(csv, index=False)
    written.append(csv)

    xlsx = out_dir / f"{stem}.xlsx"
    with pd.ExcelWriter(xlsx) as xw:
        scored.to_excel(xw, sheet_name="data", index=False)
        pd.DataFrame([{"variable": it.id, "label": it.label, "type": it.type, "options": ", ".join(str(o.value) for o in it.options)} for it in template.items]).to_excel(xw, sheet_name="codebook", index=False)
    written.append(xlsx)

    try:
        import pyreadstat

        sav_df = scored.copy()
        for c in sav_df.columns:
            if c != "sheet_id":
                sav_df[c] = pd.to_numeric(sav_df[c], errors="coerce")
        labels = {it.id: it.label[:120] for it in template.items}
        sav = out_dir / f"{stem}.sav"
        pyreadstat.write_sav(sav_df, str(sav), column_labels=[labels.get(c, c) for c in sav_df.columns])
        written.append(sav)
    except ImportError:
        pass
    return written
