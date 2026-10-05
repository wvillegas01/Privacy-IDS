"""Etapa 02 - Limpieza de CICIDS2017 raw (GeneratedLabelledFlows):
- elimina las 288,602 filas en blanco (artefacto conocido, ver etapa 01)
- elimina duplicados exactos remanentes
- normaliza el nombre de las clases "Web Attack ..." (problema de encoding cp1252/latin1)
- castea inf a NaN y luego imputa/filtra
- separa identificadores (para escenario 2) del resto de features
- guarda train/test split estratificado (80/20, seed fija) en Documents (heavy output)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_io import CICIDS_IDENTIFIER_COLS, CICIDS_RAW_DIR, HEAVY_OUTPUT_DIR, RANDOM_SEED

STAGE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = STAGE_DIR / "02_outputs"
LOG_DIR = STAGE_DIR / "04_logs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

HEAVY_CICIDS_DIR = HEAVY_OUTPUT_DIR / "cicids2017"
HEAVY_CICIDS_DIR.mkdir(parents=True, exist_ok=True)


def normalize_label(raw_label: str) -> str:
    s = str(raw_label).strip()
    if s.startswith("Web Attack"):
        if "Brute Force" in s:
            return "Web Attack - Brute Force"
        if "XSS" in s:
            return "Web Attack - XSS"
        if "Sql Injection" in s or "SQL Injection" in s:
            return "Web Attack - Sql Injection"
        return "Web Attack - Other"
    return s


def main():
    log_path = LOG_DIR / "01_clean_cicids2017_log.txt"
    with open(log_path, "w", encoding="utf-8") as fh:
        def log(msg):
            print(msg.encode("ascii", "replace").decode("ascii"))
            fh.write(msg + "\n")

        files = sorted(CICIDS_RAW_DIR.glob("*.csv"))
        frames = []
        for f in files:
            # cp1252 decodes the en-dash byte (0x96) correctly in the Web Attack labels;
            # latin1 (used in stage 01 inventory) mangles it but doesn't crash on read.
            d = pd.read_csv(f, low_memory=False, encoding="cp1252")
            d.columns = [c.strip() for c in d.columns]
            frames.append(d)
        df = pd.concat(frames, ignore_index=True)
        log(f"Loaded combined raw shape = {df.shape}")

        label_col = "Label" if "Label" in df.columns else [c for c in df.columns if c.lower() == "label"][0]

        n_before = len(df)
        df = df.dropna(subset=[label_col]).copy()
        n_after_blank = len(df)
        log(f"Dropped {n_before - n_after_blank} fully-blank rows (label missing) -> {n_after_blank} rows")

        dup_count = int(df.duplicated().sum())
        df = df.drop_duplicates().copy()
        log(f"Dropped {dup_count} exact duplicate rows -> {len(df)} rows")

        df[label_col] = df[label_col].map(normalize_label)
        log("Label distribution after normalization:")
        for cls, cnt in df[label_col].value_counts().items():
            log(f"    {cls}: {cnt} ({cnt/len(df):.4%})")

        # inf -> NaN, then drop rows with NaN in numeric feature columns (small: ~4.3k cells)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        n_before_nan = len(df)
        df = df.dropna(subset=numeric_cols).copy()
        log(f"Dropped {n_before_nan - len(df)} rows with inf/NaN in numeric features -> {len(df)} rows")

        identifier_cols = [c for c in CICIDS_IDENTIFIER_COLS if c in df.columns]
        log(f"Identifier columns (scenario 2 target): {identifier_cols}")

        df["binary_label"] = (df[label_col] != "BENIGN").astype(int)
        log("Binary label distribution:")
        for cls, cnt in df["binary_label"].value_counts().items():
            log(f"    {cls}: {cnt} ({cnt/len(df):.4%})")

        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=RANDOM_SEED, stratify=df["binary_label"]
        )
        log(f"Train shape = {train_df.shape}, test shape = {test_df.shape}")

        train_path = HEAVY_CICIDS_DIR / "cicids2017_train.parquet"
        test_path = HEAVY_CICIDS_DIR / "cicids2017_test.parquet"
        train_df.to_parquet(train_path, index=False)
        test_df.to_parquet(test_path, index=False)
        log(f"Saved: {train_path}")
        log(f"Saved: {test_path}")

        feature_cols = [c for c in df.columns if c not in identifier_cols + [label_col, "binary_label"]]
        taxonomy = {
            "dataset": "CICIDS2017",
            "label_col": label_col,
            "binary_label_col": "binary_label",
            "identifier_cols": identifier_cols,
            "feature_cols": feature_cols,
            "n_rows_final": len(df),
            "n_train": len(train_df),
            "n_test": len(test_df),
            "train_path": str(train_path),
            "test_path": str(test_path),
        }
        with open(OUT_DIR / "cicids2017_taxonomy.json", "w", encoding="utf-8") as f:
            json.dump(taxonomy, f, indent=2)
        log("Done.")


if __name__ == "__main__":
    main()
