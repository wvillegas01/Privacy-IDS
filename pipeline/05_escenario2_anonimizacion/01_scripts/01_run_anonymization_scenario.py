"""Etapa 05 - Escenario 2: anonimizacion de identificadores de red.

Los identificadores crudos (ver etapa 04, `naive_with_raw_ids`) se
generalizan (ver `anonymization_module.py`: IP -> prefijo /16, puerto ->
banda IANA) en vez de eliminarse. Se reentrena deteccion con esas versiones
generalizadas y se vuelve a correr el ataque de re-identificacion con la
MISMA evidencia generalizada (no con comportamiento puro -- ese numero ya
esta en la etapa 03 y sirve como referencia de piso, no como el resultado
de este escenario).

Comparaciones:
  - Utilidad: naive_with_raw_ids (etapa 04) vs. anonymized (esta etapa).
  - Privacidad: ataque naive (etapa 04) vs. ataque con evidencia
    generalizada (esta etapa) vs. ataque sin ningun identificador (etapa 03,
    piso teorico).
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "03_ataque_reidentificacion" / "01_scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "04_escenario1_baseline" / "01_scripts"))

from anonymization_module import (  # noqa: E402
    CICIDS_RAW_ID_FEATURES,
    UNSW_RAW_ID_FEATURES,
    generalize_ip,
    generalize_port,
)
from attack_module import build_cicids_target, run_attack  # noqa: E402
from detection_module import train_and_evaluate_all  # noqa: E402

STAGE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = STAGE_DIR / "02_outputs"
LOG_DIR = STAGE_DIR / "04_logs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

PREPROC_OUT = STAGE_DIR.parent / "02_preprocesamiento_baseline" / "02_outputs"
STAGE03_OUT = STAGE_DIR.parent / "03_ataque_reidentificacion" / "02_outputs"
STAGE04_OUT = STAGE_DIR.parent / "04_escenario1_baseline" / "02_outputs"


def anonymize_columns(df: pd.DataFrame, dataset: str) -> tuple[pd.DataFrame, list]:
    df = df.copy()
    if dataset == "CICIDS2017":
        ip_cols = ["Source IP", "Destination IP"]
        port_cols = ["Source Port", "Destination Port"]
    else:
        ip_cols = ["srcip", "dstip"]
        port_cols = ["sport", "dsport"]

    anon_cols = []
    for c in ip_cols:
        new_c = f"{c}_anon"
        df[new_c] = generalize_ip(df[c])
        anon_cols.append(new_c)
    for c in port_cols:
        new_c = f"{c}_anon"
        df[new_c] = generalize_port(df[c])
        anon_cols.append(new_c)
    return df, anon_cols


def main():
    log_path = LOG_DIR / "01_run_anonymization_scenario_log.txt"
    with open(log_path, "w", encoding="utf-8") as fh:
        def log(msg):
            print(msg)
            fh.write(msg + "\n")

        with open(STAGE03_OUT / "baseline_attack_results.json") as f:
            stage03_results = json.load(f)
        with open(STAGE04_OUT / "scenario1_baseline_results.json") as f:
            stage04_results = json.load(f)

        all_results = {}
        for dataset_key, taxonomy_file, target_col in [
            ("CICIDS2017", "cicids2017_taxonomy.json", "Source IP"),
            ("UNSW-NB15", "unsw_nb15_taxonomy.json", "srcip"),
        ]:
            with open(PREPROC_OUT / taxonomy_file) as f:
                tax = json.load(f)

            train_df = pd.read_parquet(tax["train_path"])
            test_df = pd.read_parquet(tax["test_path"])
            behavioral_cols = tax["feature_cols"]

            train_anon, anon_cols = anonymize_columns(train_df, dataset_key)
            test_anon, _ = anonymize_columns(test_df, dataset_key)
            anon_feature_cols = behavioral_cols + anon_cols
            log(f"[{dataset_key}] anonymized identifier columns: {anon_cols}")
            for c in anon_cols:
                log(f"  '{c}': n_unique(train)={train_anon[c].nunique()}")

            log(f"[{dataset_key}] -- detection: anonymized --")
            det_anon = train_and_evaluate_all(train_anon, test_anon, anon_feature_cols, tax["binary_label_col"])
            for model_name, metrics in det_anon.items():
                log(f"[{dataset_key}][anonymized][{model_name}] " + ", ".join(
                    f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()
                ))

            log(f"[{dataset_key}] -- privacy attack: evidence = behavioral + anonymized ids --")
            if dataset_key == "CICIDS2017":
                target_train, target_test = build_cicids_target(train_df, test_df, top_n=20, id_col=target_col)
            else:
                target_train, target_test = train_df[target_col], test_df[target_col]
            attack_anon = run_attack(train_anon, test_anon, anon_feature_cols, target_train, target_test)
            log(f"[{dataset_key}][attack_anonymized] top1={attack_anon.top1_accuracy:.4f}, "
                f"top5={attack_anon.top5_accuracy:.4f}, majority={attack_anon.majority_baseline_accuracy:.4f}")

            naive_top1 = stage04_results[dataset_key]["privacy_risk_scenario1_naive"]["top1_accuracy"]
            floor_top1 = stage03_results[dataset_key]["top1_accuracy"]
            log(f"[{dataset_key}] risk comparison: naive(S1)={naive_top1:.4f} -> "
                f"anonymized(S2)={attack_anon.top1_accuracy:.4f} -> "
                f"no-identifiers-floor={floor_top1:.4f}")

            naive_f1_xgb = stage04_results[dataset_key]["detection_naive_with_raw_ids"]["xgboost"]["f1"]
            anon_f1_xgb = det_anon["xgboost"]["f1"]
            log(f"[{dataset_key}] utility comparison (XGBoost F1): naive(S1)={naive_f1_xgb:.4f} -> "
                f"anonymized(S2)={anon_f1_xgb:.4f} (delta={anon_f1_xgb - naive_f1_xgb:+.4f})")

            all_results[dataset_key] = {
                "detection_anonymized": det_anon,
                "privacy_risk_anonymized": {
                    "top1_accuracy": attack_anon.top1_accuracy,
                    "top5_accuracy": attack_anon.top5_accuracy,
                    "majority_baseline_accuracy": attack_anon.majority_baseline_accuracy,
                },
                "comparison_to_scenario1_naive": {
                    "risk_top1_naive": naive_top1,
                    "risk_top1_anonymized": attack_anon.top1_accuracy,
                    "risk_top1_no_identifiers_floor": floor_top1,
                    "utility_f1_xgboost_naive": naive_f1_xgb,
                    "utility_f1_xgboost_anonymized": anon_f1_xgb,
                },
            }

        with open(OUT_DIR / "scenario2_anonymization_results.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        log("\nDone. Results saved.")


if __name__ == "__main__":
    main()
