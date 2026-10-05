"""Analisis de sensibilidad del umbral de importancia acumulada de S3 (etapa
06), pedido por el usuario: en vez de solo 50%, probar 25% y 75% para no
presentar S3 como "la tecnica mas eficiente" apoyado en un unico punto de
diseno arbitrario. XGBoost unicamente, seed=42, igual que el resto de la
comparacion transversal.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "03_ataque_reidentificacion" / "01_scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "04_escenario1_baseline" / "01_scripts"))

from attack_module import build_cicids_target, run_attack  # noqa: E402
from detection_module import train_and_evaluate_all  # noqa: E402

STAGE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = STAGE_DIR / "02_outputs"
LOG_DIR = STAGE_DIR / "04_logs"
PREPROC_OUT = STAGE_DIR.parent / "02_preprocesamiento_baseline" / "02_outputs"
ATTACK_OUT = STAGE_DIR.parent / "03_ataque_reidentificacion" / "02_outputs"
THRESHOLDS = [0.25, 0.50, 0.75]


def select_features_to_remove(ranking, cutoff):
    total = sum(v for _, v in ranking)
    cum = 0.0
    removed = []
    for feat, imp in ranking:
        if cum / total >= cutoff:
            break
        removed.append(feat)
        cum += imp
    return removed, cum / total


def main():
    log_path = LOG_DIR / "02_threshold_sensitivity_log.txt"
    with open(log_path, "w", encoding="utf-8") as fh:
        def log(msg):
            print(msg)
            fh.write(msg + "\n")

        with open(ATTACK_OUT / "baseline_attack_results.json") as f:
            attack_results = json.load(f)

        results = {}
        for dataset_key, taxonomy_file, target_col in [
            ("CICIDS2017", "cicids2017_taxonomy.json", "Source IP"),
            ("UNSW-NB15", "unsw_nb15_taxonomy.json", "srcip"),
        ]:
            with open(PREPROC_OUT / taxonomy_file) as f:
                tax = json.load(f)
            train_df = pd.read_parquet(tax["train_path"])
            test_df = pd.read_parquet(tax["test_path"])
            behavioral_cols = tax["feature_cols"]
            ranking = attack_results[dataset_key]["feature_importance_ranking"]

            if dataset_key == "CICIDS2017":
                target_train, target_test = build_cicids_target(train_df, test_df, top_n=20, id_col=target_col)
            else:
                target_train, target_test = train_df[target_col], test_df[target_col]

            dataset_results = []
            for thr in THRESHOLDS:
                removed, achieved = select_features_to_remove(ranking, thr)
                reduced_cols = [c for c in behavioral_cols if c not in removed]
                det = train_and_evaluate_all(train_df, test_df, reduced_cols, tax["binary_label_col"],
                                              model_names=["xgboost"])
                att = run_attack(train_df, test_df, reduced_cols, target_train, target_test)
                log(f"[{dataset_key}][thr={thr}] removed={len(removed)}/{len(behavioral_cols)} "
                    f"(achieved={achieved:.3f}) F1={det['xgboost']['f1']:.4f}, risk={att.top1_accuracy:.4f}")
                dataset_results.append({
                    "threshold": thr, "n_removed": len(removed), "achieved_cumulative": achieved,
                    "f1": det["xgboost"]["f1"], "risk": att.top1_accuracy,
                })
            results[dataset_key] = dataset_results

        with open(OUT_DIR / "threshold_sensitivity.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        log("\nDone.")


if __name__ == "__main__":
    main()
