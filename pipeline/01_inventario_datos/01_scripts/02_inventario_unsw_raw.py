"""Etapa 01 (extension) - Inventario de UNSW-NB15 RAW (4 CSV sin encabezado,
con srcip/sport/dstip/dsport reales). Reemplaza a la variante ML-ready como
fuente para el Escenario 2 (anonimizacion), ver decision 2026-09-11.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

STAGE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = STAGE_DIR / "02_outputs"
LOG_DIR = STAGE_DIR / "04_logs"

RAW_DIR = Path(os.environ.get("UNSW_NB15_RAW_DIR", "data/UNSW-NB15_raw"))
FEATURES_CSV = RAW_DIR / "NUSW-NB15_features.csv"

COLUMN_NAME_FIX = {
    "smeansz": "smean",
    "dmeansz": "dmean",
    "res_bdy_len": "response_body_len",
    "ct_src_ ltm": "ct_src_ltm",
}


def load_column_names():
    feat = pd.read_csv(FEATURES_CSV, encoding="latin1")
    names = [str(n).strip() for n in feat["Name"].tolist()]
    names = [COLUMN_NAME_FIX.get(n, n) for n in names]
    return names


def main():
    columns = load_column_names()
    assert len(columns) == 49, f"expected 49 columns, got {len(columns)}"

    files = sorted(RAW_DIR.glob("UNSW-NB15_[1-4].csv"))
    log_path = LOG_DIR / "02_inventario_unsw_raw_log.txt"
    with open(log_path, "w", encoding="utf-8") as fh:
        def log(msg):
            print(msg)
            fh.write(msg + "\n")

        log(f"Files: {[f.name for f in files]}")
        frames = []
        for f in files:
            d = pd.read_csv(f, header=None, names=columns, low_memory=False, encoding="latin1")
            frames.append(d)
            log(f"  loaded {f.name}: {d.shape}")
        df = pd.concat(frames, ignore_index=True)
        log(f"Combined shape = {df.shape}")

        dup_count = int(df.duplicated().sum())
        log(f"Exact duplicate rows = {dup_count} ({dup_count/len(df):.2%})")

        df["attack_cat"] = df["attack_cat"].astype(str).str.strip()
        df.loc[df["attack_cat"].isin(["nan", "", "-"]), "attack_cat"] = "Normal"
        # collapse known label variants (dataset has some inconsistent spacing/case)
        norm_map = {}
        for v in df["attack_cat"].unique():
            norm_map[v] = v.strip()
        df["attack_cat"] = df["attack_cat"].map(norm_map)

        log("Label (0/1) distribution:")
        for cls, cnt in df["Label"].value_counts().items():
            log(f"    {cls}: {cnt} ({cnt/len(df):.4%})")

        log("attack_cat distribution:")
        for cls, cnt in df["attack_cat"].value_counts().items():
            log(f"    {cls}: {cnt} ({cnt/len(df):.4%})")

        identifier_cols = ["srcip", "sport", "dstip", "dsport"]
        for c in identifier_cols:
            n_unique = df[c].nunique()
            n_missing = df[c].isna().sum()
            log(f"identifier col '{c}': n_unique={n_unique}, n_missing={n_missing}")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        nan_counts = {c: int(df[c].isna().sum()) for c in numeric_cols}
        cols_with_nan = {k: v for k, v in nan_counts.items() if v > 0}
        log(f"columns with NaN: {cols_with_nan}")

        profile = {
            "name": "UNSW-NB15-raw",
            "n_rows": len(df),
            "n_cols": df.shape[1],
            "duplicate_rows": dup_count,
            "duplicate_pct": dup_count / len(df),
            "label_distribution": {str(k): int(v) for k, v in df["Label"].value_counts().items()},
            "attack_cat_distribution": {str(k): int(v) for k, v in df["attack_cat"].value_counts().items()},
            "identifier_cardinality": {c: int(df[c].nunique()) for c in identifier_cols},
            "cols_with_nan": cols_with_nan,
        }
        with open(OUT_DIR / "unsw_nb15_raw_profile.json", "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2)

        log(f"\nDone. Profile saved.")


if __name__ == "__main__":
    main()
