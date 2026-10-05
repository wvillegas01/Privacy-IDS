"""Etapa 02 - Limpieza de UNSW-NB15 raw (UNSW-NB15_1..4.csv):
- asigna encabezados desde NUSW-NB15_features.csv
- elimina duplicados exactos (18.92% detectado en etapa 01)
- normaliza attack_cat (Backdoor/Backdoors -> Backdoor; vacio/"-" -> Normal)
- imputa ct_flw_http_mthd / is_ftp_login a 0 (no aplica en flujos no-HTTP/FTP)
- separa identificadores (para escenario 2) del resto de features
- guarda train/test split estratificado (80/20, seed fija) en Documents (heavy output)
"""
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_io import (
    HEAVY_OUTPUT_DIR,
    RANDOM_SEED,
    UNSW_COLUMN_NAME_FIX,
    UNSW_FEATURES_CSV,
    UNSW_IDENTIFIER_COLS,
    UNSW_RAW_DIR,
)

STAGE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = STAGE_DIR / "02_outputs"
LOG_DIR = STAGE_DIR / "04_logs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

HEAVY_UNSW_DIR = HEAVY_OUTPUT_DIR / "unsw_nb15"
HEAVY_UNSW_DIR.mkdir(parents=True, exist_ok=True)


def load_column_names():
    feat = pd.read_csv(UNSW_FEATURES_CSV, encoding="latin1")
    names = [str(n).strip() for n in feat["Name"].tolist()]
    names = [UNSW_COLUMN_NAME_FIX.get(n, n) for n in names]
    return names


def main():
    log_path = LOG_DIR / "02_clean_unsw_nb15_log.txt"
    with open(log_path, "w", encoding="utf-8") as fh:
        def log(msg):
            print(msg)
            fh.write(msg + "\n")

        columns = load_column_names()
        assert len(columns) == 49

        files = sorted(UNSW_RAW_DIR.glob("UNSW-NB15_[1-4].csv"))
        frames = []
        for f in files:
            d = pd.read_csv(f, header=None, names=columns, low_memory=False, encoding="latin1")
            frames.append(d)
        df = pd.concat(frames, ignore_index=True)
        log(f"Loaded combined raw shape = {df.shape}")

        dup_count = int(df.duplicated().sum())
        df = df.drop_duplicates().copy()
        log(f"Dropped {dup_count} exact duplicate rows -> {len(df)} rows")

        df["attack_cat"] = df["attack_cat"].astype(str).str.strip()
        df.loc[df["attack_cat"].isin(["nan", "", "-"]), "attack_cat"] = "Normal"
        df["attack_cat"] = df["attack_cat"].replace({"Backdoors": "Backdoor"})
        log("attack_cat distribution after normalization:")
        for cls, cnt in df["attack_cat"].value_counts().items():
            log(f"    {cls}: {cnt} ({cnt/len(df):.4%})")

        for c in ["ct_flw_http_mthd", "is_ftp_login"]:
            n_missing = int(df[c].isna().sum())
            df[c] = df[c].fillna(0)
            log(f"Imputed {n_missing} missing values in '{c}' to 0")

        # sport/dsport: a handful of rows (~312 out of 2M+) store the port as a hex
        # string (e.g. '0x000c') or '-' instead of a decimal integer -- known raw-file
        # quirk. Parse hex, coerce the rest, sentinel -1 for anything unparseable.
        def parse_port(v):
            v = str(v).strip()
            if v.lower().startswith("0x"):
                try:
                    return int(v, 16)
                except ValueError:
                    return None
            try:
                return int(v)
            except ValueError:
                return None

        for c in ["sport", "dsport"]:
            s = df[c].astype(str).str.strip()
            n_hex = int(s.str.match(r"^0[xX][0-9a-fA-F]+$").sum())
            parsed = s.map(parse_port)
            n_unparsed = int(parsed.isna().sum())
            df[c] = parsed.fillna(-1).astype("int64")
            log(f"'{c}': parsed {n_hex} hex-format values, sentineled {n_unparsed} unparseable to -1")

        # generic safety net: every column declared integer/Float/Binary in the
        # official feature list must end up numeric. A handful of them (e.g.
        # ct_ftp_cmd) mix blank-string placeholders with ints in the raw CSVs,
        # which forces object dtype and breaks parquet serialization.
        numeric_like_cols = [
            c for c in [
                "dur", "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss",
                "sload", "dload", "spkts", "dpkts", "swin", "dwin", "stcpb",
                "dtcpb", "smean", "dmean", "trans_depth", "response_body_len",
                "sjit", "djit", "stime", "ltime", "sinpkt", "dinpkt", "tcprtt",
                "synack", "ackdat", "is_sm_ips_ports", "ct_state_ttl",
                "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd", "ct_srv_src",
                "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm",
                "ct_dst_sport_ltm", "ct_dst_src_ltm",
            ]
            if c in df.columns and df[c].dtype == object
        ]
        for c in numeric_like_cols:
            before_na = int(df[c].isna().sum())
            df[c] = pd.to_numeric(df[c], errors="coerce")
            after_na = int(df[c].isna().sum())
            df[c] = df[c].fillna(0)
            log(f"Coerced object column '{c}' to numeric ({after_na - before_na} new NaN -> filled 0)")

        df["Label"] = df["Label"].astype(int)
        log("Label (binary) distribution:")
        for cls, cnt in df["Label"].value_counts().items():
            log(f"    {cls}: {cnt} ({cnt/len(df):.4%})")

        identifier_cols = [c for c in UNSW_IDENTIFIER_COLS if c in df.columns]
        log(f"Identifier columns (scenario 2 target): {identifier_cols}")

        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=RANDOM_SEED, stratify=df["Label"]
        )
        log(f"Train shape = {train_df.shape}, test shape = {test_df.shape}")

        train_path = HEAVY_UNSW_DIR / "unsw_nb15_train.parquet"
        test_path = HEAVY_UNSW_DIR / "unsw_nb15_test.parquet"
        train_df.to_parquet(train_path, index=False)
        test_df.to_parquet(test_path, index=False)
        log(f"Saved: {train_path}")
        log(f"Saved: {test_path}")

        feature_cols = [c for c in df.columns if c not in identifier_cols + ["attack_cat", "Label"]]
        taxonomy = {
            "dataset": "UNSW-NB15",
            "label_col": "attack_cat",
            "binary_label_col": "Label",
            "identifier_cols": identifier_cols,
            "feature_cols": feature_cols,
            "n_rows_final": len(df),
            "n_train": len(train_df),
            "n_test": len(test_df),
            "train_path": str(train_path),
            "test_path": str(test_path),
        }
        with open(OUT_DIR / "unsw_nb15_taxonomy.json", "w", encoding="utf-8") as f:
            json.dump(taxonomy, f, indent=2)
        log("Done.")


if __name__ == "__main__":
    main()
