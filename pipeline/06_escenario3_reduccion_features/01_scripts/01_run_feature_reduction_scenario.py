"""Etapa 06 - Escenario 3: reduccion de features sensibles.

Criterio de "sensible" (no arbitrario): el ranking de importancia del
ataque de re-identificacion de la etapa 03. Se elimina el conjunto minimo
de features de mayor importancia necesario para cubrir el 50% de la
importancia acumulada del ataque -- distinto por diseno del Escenario 2
(que generaliza columnas IDENTIFICADOR, nunca features de comportamiento).

Se reentrena deteccion con el feature set reducido y se vuelve a correr el
ataque (evidencia = features de comportamiento reducidas, sin identificador
alguno, igual que en la etapa 03) para medir si esta tecnica logra reducir
el riesgo donde la anonimizacion de identificadores (etapa 05) no lo logro
en CICIDS2017.
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
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

PREPROC_OUT = STAGE_DIR.parent / "02_preprocesamiento_baseline" / "02_outputs"
ATTACK_OUT = STAGE_DIR.parent / "03_ataque_reidentificacion" / "02_outputs"
CUMULATIVE_IMPORTANCE_CUTOFF = 0.5


def select_features_to_remove(ranking, cutoff=CUMULATIVE_IMPORTANCE_CUTOFF):
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
    log_path = LOG_DIR / "01_run_feature_reduction_scenario_log.txt"
    with open(log_path, "w", encoding="utf-8") as fh:
        def log(msg):
            print(msg)
            fh.write(msg + "\n")

        with open(ATTACK_OUT / "baseline_attack_results.json") as f:
            attack_results = json.load(f)

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

            ranking = attack_results[dataset_key]["feature_importance_ranking"]
            removed_feats, achieved_cum = select_features_to_remove(ranking)
            reduced_cols = [c for c in behavioral_cols if c not in removed_feats]
            log(f"[{dataset_key}] removing {len(removed_feats)}/{len(behavioral_cols)} features "
                f"(covers {achieved_cum:.1%} of re-id importance): {removed_feats}")
            log(f"[{dataset_key}] remaining features: {len(reduced_cols)}")

            log(f"[{dataset_key}] -- detection: feature-reduced --")
            det_reduced = train_and_evaluate_all(train_df, test_df, reduced_cols, tax["binary_label_col"])
            for model_name, metrics in det_reduced.items():
                log(f"[{dataset_key}][reduced][{model_name}] " + ", ".join(
                    f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()
                ))

            log(f"[{dataset_key}] -- privacy attack: evidence = reduced behavioral features --")
            if dataset_key == "CICIDS2017":
                target_train, target_test = build_cicids_target(train_df, test_df, top_n=20, id_col=target_col)
            else:
                target_train, target_test = train_df[target_col], test_df[target_col]
            attack_reduced = run_attack(train_df, test_df, reduced_cols, target_train, target_test)
            log(f"[{dataset_key}][attack_reduced] top1={attack_reduced.top1_accuracy:.4f}, "
                f"top5={attack_reduced.top5_accuracy:.4f}, majority={attack_reduced.majority_baseline_accuracy:.4f}")

            behavioral_only_top1 = attack_results[dataset_key]["top1_accuracy"]
            log(f"[{dataset_key}] risk comparison: full-behavioral(etapa03)={behavioral_only_top1:.4f} -> "
                f"reduced(S3)={attack_reduced.top1_accuracy:.4f}")

            all_results[dataset_key] = {
                "removed_features": removed_feats,
                "achieved_cumulative_importance": achieved_cum,
                "remaining_feature_count": len(reduced_cols),
                "detection_reduced": det_reduced,
                "privacy_risk_reduced": {
                    "top1_accuracy": attack_reduced.top1_accuracy,
                    "top5_accuracy": attack_reduced.top5_accuracy,
                    "majority_baseline_accuracy": attack_reduced.majority_baseline_accuracy,
                },
                "risk_before_reduction_behavioral_only": behavioral_only_top1,
            }

        with open(OUT_DIR / "scenario3_feature_reduction_results.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        log("\nDone. Results saved.")


if __name__ == "__main__":
    main()
