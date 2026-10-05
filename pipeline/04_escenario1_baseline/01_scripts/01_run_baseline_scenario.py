"""Etapa 04 - Escenario 1 (baseline = "datos sin modificar").

Correccion de diseno 2026-09-14: el baseline debe incluir los
identificadores de red CRUDOS como features de deteccion (Source/Destination
IP y Puerto -- no Flow ID/Timestamp, bookkeeping puro sin valor predictivo)
para que el Escenario 2 (anonimizacion) tenga algo real que anonimizar y un
costo de utilidad medible. Se reportan DOS variantes de deteccion:
  - "behavioral_only": las mismas features de siempre (sin identificadores)
    -- practica estandar en literatura de IDS, sirve de referencia interna.
  - "naive_with_raw_ids": features de comportamiento + identificadores
    crudos -- esta es la version que representa "unmodified benchmark data"
    tal como se prometio en el resumen enviado al editor, y la que se usa
    para comparar contra el Escenario 2.

Tambien corre el ataque de re-identificacion "naive" (evidencia = features
de comportamiento + identificadores crudos, EXCLUYENDO solo la columna
objetivo) -- esta es la referencia de riesgo real del Escenario 1. El
resultado ya calculado en la etapa 03 (comportamiento puro, sin ningun
identificador) pasa a representar el riesgo POST-anonimizacion (Escenario 2,
etapa 05), no una linea base generica.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "05_escenario2_anonimizacion" / "01_scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "03_ataque_reidentificacion" / "01_scripts"))

from anonymization_module import CICIDS_RAW_ID_FEATURES, UNSW_RAW_ID_FEATURES  # noqa: E402
from attack_module import build_cicids_target, run_attack  # noqa: E402
from detection_module import train_and_evaluate_all  # noqa: E402

STAGE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = STAGE_DIR / "02_outputs"
LOG_DIR = STAGE_DIR / "04_logs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

PREPROC_OUT = STAGE_DIR.parent / "02_preprocesamiento_baseline" / "02_outputs"


def main():
    log_path = LOG_DIR / "01_run_baseline_scenario_log.txt"
    with open(log_path, "w", encoding="utf-8") as fh:
        def log(msg):
            print(msg)
            fh.write(msg + "\n")

        all_results = {}
        for dataset_key, taxonomy_file, raw_id_cols, target_col in [
            ("CICIDS2017", "cicids2017_taxonomy.json", CICIDS_RAW_ID_FEATURES, "Source IP"),
            ("UNSW-NB15", "unsw_nb15_taxonomy.json", UNSW_RAW_ID_FEATURES, "srcip"),
        ]:
            with open(PREPROC_OUT / taxonomy_file) as f:
                tax = json.load(f)

            train_df = pd.read_parquet(tax["train_path"])
            test_df = pd.read_parquet(tax["test_path"])
            behavioral_cols = tax["feature_cols"]
            naive_cols = behavioral_cols + raw_id_cols
            log(f"[{dataset_key}] train={train_df.shape}, test={test_df.shape}, "
                f"n_behavioral={len(behavioral_cols)}, n_naive={len(naive_cols)}")

            log(f"[{dataset_key}] -- detection: behavioral_only --")
            det_behavioral = train_and_evaluate_all(train_df, test_df, behavioral_cols, tax["binary_label_col"])
            for model_name, metrics in det_behavioral.items():
                log(f"[{dataset_key}][behavioral_only][{model_name}] " + ", ".join(
                    f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()
                ))

            log(f"[{dataset_key}] -- detection: naive_with_raw_ids --")
            det_naive = train_and_evaluate_all(train_df, test_df, naive_cols, tax["binary_label_col"])
            for model_name, metrics in det_naive.items():
                log(f"[{dataset_key}][naive_with_raw_ids][{model_name}] " + ", ".join(
                    f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()
                ))

            log(f"[{dataset_key}] -- privacy attack: naive (raw ids as evidence, target excluded) --")
            attack_evidence_cols = [c for c in naive_cols if c != target_col]
            if dataset_key == "CICIDS2017":
                target_train, target_test = build_cicids_target(train_df, test_df, top_n=20, id_col=target_col)
            else:
                target_train, target_test = train_df[target_col], test_df[target_col]
            attack_naive = run_attack(train_df, test_df, attack_evidence_cols, target_train, target_test)
            log(f"[{dataset_key}][attack_naive] top1={attack_naive.top1_accuracy:.4f}, "
                f"top5={attack_naive.top5_accuracy:.4f}, majority={attack_naive.majority_baseline_accuracy:.4f}, "
                f"n_classes={attack_naive.n_classes}")

            all_results[dataset_key] = {
                "detection_behavioral_only": det_behavioral,
                "detection_naive_with_raw_ids": det_naive,
                "privacy_risk_scenario1_naive": {
                    "top1_accuracy": attack_naive.top1_accuracy,
                    "top5_accuracy": attack_naive.top5_accuracy,
                    "majority_baseline_accuracy": attack_naive.majority_baseline_accuracy,
                    "n_classes": attack_naive.n_classes,
                },
            }

        with open(OUT_DIR / "scenario1_baseline_results.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        log("\nDone. Results saved.")


if __name__ == "__main__":
    main()
