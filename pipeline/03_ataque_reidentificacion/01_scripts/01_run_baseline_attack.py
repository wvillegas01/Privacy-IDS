"""Etapa 03 - Corre el ataque de re-identificacion (ver attack_module.py)
sobre las features SIN transformar (linea base) de ambos datasets. Produce:
  - la tasa de exito del ataque como referencia de riesgo superior (usada
    tambien como comparacion en escenarios 2/3/4)
  - el ranking de importancia de features para el ataque, que define el
    criterio de "feature sensible" para el Escenario 3 (etapa 06)
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from attack_module import build_cicids_target, run_attack

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "02_preprocesamiento_baseline" / "01_scripts"))
from common_io import HEAVY_OUTPUT_DIR

STAGE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = STAGE_DIR / "02_outputs"
LOG_DIR = STAGE_DIR / "04_logs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

PREV_STAGE_OUT = STAGE_DIR.parent / "02_preprocesamiento_baseline" / "02_outputs"


def log_result(fh, log, name, result):
    log(f"[{name}] n_classes={result.n_classes}, n_train_used={result.n_train_used}, n_test={result.n_test}")
    log(f"[{name}] top1_accuracy={result.top1_accuracy:.4f}, top5_accuracy={result.top5_accuracy:.4f}, "
        f"majority_baseline={result.majority_baseline_accuracy:.4f}")
    lift = result.top1_accuracy / result.majority_baseline_accuracy if result.majority_baseline_accuracy > 0 else float("nan")
    log(f"[{name}] lift over chance = {lift:.2f}x")
    top_feats = sorted(result.feature_importances.items(), key=lambda kv: -kv[1])[:15]
    log(f"[{name}] top-15 features driving re-identification:")
    for feat, imp in top_feats:
        log(f"    {feat}: {imp:.4f}")


def main():
    log_path = LOG_DIR / "01_run_baseline_attack_log.txt"
    with open(log_path, "w", encoding="utf-8") as fh:
        def log(msg):
            print(msg)
            fh.write(msg + "\n")

        # --- CICIDS2017 ---
        with open(PREV_STAGE_OUT / "cicids2017_taxonomy.json") as f:
            cicids_tax = json.load(f)
        cicids_train = pd.read_parquet(cicids_tax["train_path"])
        cicids_test = pd.read_parquet(cicids_tax["test_path"])
        target_train, target_test = build_cicids_target(cicids_train, cicids_test, top_n=20, id_col="Source IP")
        log(f"[CICIDS2017] target classes (top-20 Source IP + Other): {target_train.nunique()}")
        cicids_result = run_attack(
            cicids_train, cicids_test, cicids_tax["feature_cols"], target_train, target_test
        )
        log_result(fh, log, "CICIDS2017", cicids_result)

        # --- UNSW-NB15 ---
        with open(PREV_STAGE_OUT / "unsw_nb15_taxonomy.json") as f:
            unsw_tax = json.load(f)
        unsw_train = pd.read_parquet(unsw_tax["train_path"])
        unsw_test = pd.read_parquet(unsw_tax["test_path"])
        log(f"[UNSW-NB15] target classes (srcip, no bucketing needed): {unsw_train['srcip'].nunique()}")
        unsw_result = run_attack(
            unsw_train, unsw_test, unsw_tax["feature_cols"], unsw_train["srcip"], unsw_test["srcip"]
        )
        log_result(fh, log, "UNSW-NB15", unsw_result)

        results = {
            "CICIDS2017": {
                "top1_accuracy": cicids_result.top1_accuracy,
                "top5_accuracy": cicids_result.top5_accuracy,
                "majority_baseline_accuracy": cicids_result.majority_baseline_accuracy,
                "n_classes": cicids_result.n_classes,
                "n_train_used": cicids_result.n_train_used,
                "n_test": cicids_result.n_test,
                "feature_importance_ranking": sorted(
                    cicids_result.feature_importances.items(), key=lambda kv: -kv[1]
                ),
            },
            "UNSW-NB15": {
                "top1_accuracy": unsw_result.top1_accuracy,
                "top5_accuracy": unsw_result.top5_accuracy,
                "majority_baseline_accuracy": unsw_result.majority_baseline_accuracy,
                "n_classes": unsw_result.n_classes,
                "n_train_used": unsw_result.n_train_used,
                "n_test": unsw_result.n_test,
                "feature_importance_ranking": sorted(
                    unsw_result.feature_importances.items(), key=lambda kv: -kv[1]
                ),
            },
        }
        with open(OUT_DIR / "baseline_attack_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        log("\nDone. Results saved.")


if __name__ == "__main__":
    main()
