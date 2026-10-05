"""Etapa 01 (extension) - Inventario de CICIDS2017 RAW (GeneratedLabelledFlows,
con Flow ID / Source IP / Source Port / Destination IP / Timestamp reales).
Reemplaza a MachineLearningCSV.zip como fuente para el Escenario 2
(anonimizacion), ver decision 2026-09-11.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

STAGE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = STAGE_DIR / "02_outputs"
LOG_DIR = STAGE_DIR / "04_logs"

RAW_DIR = Path(os.environ.get("CICIDS2017_RAW_DIR", "data/CICIDS2017_raw/TrafficLabelling"))


def main():
    files = sorted(RAW_DIR.glob("*.csv"))
    log_path = LOG_DIR / "03_inventario_cicids_raw_log.txt"
    with open(log_path, "w", encoding="utf-8") as fh:
        def log(msg):
            print(msg.encode("ascii", "replace").decode("ascii"))
            fh.write(msg + "\n")

        log(f"Files: {[f.name for f in files]}")
        frames = []
        for f in files:
            d = pd.read_csv(f, low_memory=False, encoding="latin1")
            d.columns = [c.strip() for c in d.columns]
            d["__source_file"] = f.name
            frames.append(d)
            log(f"  loaded {f.name}: {d.shape}")
        df = pd.concat(frames, ignore_index=True)
        log(f"Combined shape = {df.shape}")

        fully_blank = int(df.isna().all(axis=1).sum())
        log(f"Fully-blank rows (all-NaN, known CICIDS2017 raw-release padding artifact) = {fully_blank} ({fully_blank/len(df):.2%})")

        dup_count = int(df.duplicated().sum())
        log(f"Exact duplicate rows (incl. blank rows counted as dupes of each other) = {dup_count} ({dup_count/len(df):.2%})")

        label_col = "Label" if "Label" in df.columns else [c for c in df.columns if c.lower() == "label"][0]
        log(f"Label distribution ({label_col}):")
        for cls, cnt in df[label_col].value_counts().items():
            log(f"    {cls}: {cnt} ({cnt/len(df):.4%})")

        identifier_cols = ["Flow ID", "Source IP", "Source Port", "Destination IP", "Destination Port", "Timestamp"]
        identifier_cols = [c for c in identifier_cols if c in df.columns]
        log(f"identifier columns present: {identifier_cols}")
        for c in identifier_cols:
            log(f"  '{c}': n_unique={df[c].nunique()}, n_missing={df[c].isna().sum()}")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        inf_counts = {}
        nan_counts = {}
        for c in numeric_cols:
            col = df[c]
            try:
                inf_counts[c] = int(np.isinf(col.to_numpy(dtype="float64", na_value=0.0)).sum())
            except Exception:
                inf_counts[c] = 0
            nan_counts[c] = int(col.isna().sum())
        total_inf = sum(inf_counts.values())
        total_nan = sum(nan_counts.values())
        log(f"total inf cells = {total_inf}, total NaN cells = {total_nan}")
        cols_with_inf = {k: v for k, v in inf_counts.items() if v > 0}
        cols_with_nan = {k: v for k, v in nan_counts.items() if v > 0}
        log(f"cols with inf (top10): {dict(list(sorted(cols_with_inf.items(), key=lambda kv:-kv[1]))[:10])}")
        log(f"cols with NaN (top10): {dict(list(sorted(cols_with_nan.items(), key=lambda kv:-kv[1]))[:10])}")

        profile = {
            "name": "CICIDS2017-raw",
            "n_rows": len(df),
            "n_cols": df.shape[1],
            "duplicate_rows": dup_count,
            "duplicate_pct": dup_count / len(df),
            "label_distribution": {str(k): int(v) for k, v in df[label_col].value_counts().items()},
            "identifier_cardinality": {c: int(df[c].nunique()) for c in identifier_cols},
            "total_inf_cells": total_inf,
            "total_nan_cells": total_nan,
            "cols_with_inf": cols_with_inf,
            "cols_with_nan": cols_with_nan,
            "columns": list(df.columns),
        }
        with open(OUT_DIR / "cicids2017_raw_profile.json", "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2)
        log("\nDone. Profile saved.")


if __name__ == "__main__":
    main()
