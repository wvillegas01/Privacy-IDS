"""Rutas y utilidades compartidas por las etapas 02+ del pipeline Privacy-IDS-Frontiers."""
import os
from pathlib import Path

CICIDS_RAW_DIR = Path(os.environ.get("CICIDS2017_RAW_DIR", "data/CICIDS2017_raw/TrafficLabelling"))
UNSW_RAW_DIR = Path(os.environ.get("UNSW_NB15_RAW_DIR", "data/UNSW-NB15_raw"))
UNSW_FEATURES_CSV = UNSW_RAW_DIR / "NUSW-NB15_features.csv"

# Outputs pesados (datasets procesados, splits, predicciones) NO van a Dropbox.
HEAVY_OUTPUT_DIR = Path(os.environ.get("PRIVACY_IDS_HEAVY_DIR", "data/heavy_outputs"))

CICIDS_IDENTIFIER_COLS = [
    "Flow ID",
    "Source IP",
    "Source Port",
    "Destination IP",
    "Destination Port",
    "Timestamp",
]

UNSW_IDENTIFIER_COLS = ["srcip", "sport", "dstip", "dsport", "Stime", "Ltime"]

UNSW_COLUMN_NAME_FIX = {
    "smeansz": "smean",
    "dmeansz": "dmean",
    "res_bdy_len": "response_body_len",
    "ct_src_ ltm": "ct_src_ltm",
}

RANDOM_SEED = 42
