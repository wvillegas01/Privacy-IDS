"""Etapa 07 - Escenario 4: perturbacion tipo Laplace inspirada en DP.

v2 (2026-09-16): reformalizado con clipping a percentiles [1,99] (sensibilidad
fijada por el mecanismo, no el rango observado) y composicion basica
explicita: el barrido ahora es sobre epsilon_total (la garantia real a nivel
de registro completo), con epsilon_per_feature = epsilon_total / d
repartido uniformemente entre las d features numericas perturbadas.
Calibrado empiricamente: epsilon_total in {5,10,25,50,100,200,400,800} cubre
el rango interesante de la curva utilidad-riesgo (ver 01_scripts, log de
calibracion). eps_total=100 se usa como punto representativo para la
verificacion de 3 modelos y para la comparacion de 5 semillas de la etapa 08.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "03_ataque_reidentificacion" / "01_scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "04_escenario1_baseline" / "01_scripts"))

from attack_module import build_cicids_target, run_attack  # noqa: E402
from detection_module import train_and_evaluate_all  # noqa: E402
from dp_module import add_laplace_noise_clipped  # noqa: E402

STAGE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = STAGE_DIR / "02_outputs"
LOG_DIR = STAGE_DIR / "04_logs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

PREPROC_OUT = STAGE_DIR.parent / "02_preprocesamiento_baseline" / "02_outputs"
EPSILON_TOTALS = [5, 10, 25, 50, 100, 200, 400, 800]
REPRESENTATIVE_EPSILON_TOTAL = 100


def main():
    log_path = LOG_DIR / "01_run_dp_scenario_log.txt"
    with open(log_path, "w", encoding="utf-8") as fh:
        def log(msg):
            print(msg)
            fh.write(msg + "\n")

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

            if dataset_key == "CICIDS2017":
                target_train, target_test = build_cicids_target(train_df, test_df, top_n=20, id_col=target_col)
            else:
                target_train, target_test = train_df[target_col], test_df[target_col]

            d_numeric = sum(1 for c in behavioral_cols if pd.api.types.is_numeric_dtype(train_df[c]))
            log(f"[{dataset_key}] d (numeric/perturbed features) = {d_numeric}")

            sweep = []
            for eps_total in EPSILON_TOTALS:
                eps_per_feature = eps_total / d_numeric
                train_noisy, test_noisy, sensitivities, d_check, eps_tot_check = add_laplace_noise_clipped(
                    train_df, test_df, behavioral_cols, eps_per_feature
                )

                det = train_and_evaluate_all(
                    train_noisy, test_noisy, behavioral_cols, tax["binary_label_col"], model_names=["xgboost"]
                )
                attack = run_attack(train_noisy, test_noisy, behavioral_cols, target_train, target_test)

                log(f"[{dataset_key}][eps_total={eps_total}] eps_per_feature={eps_per_feature:.4f} "
                    f"xgboost f1={det['xgboost']['f1']:.4f}, auc={det['xgboost']['auc_roc']:.4f}, "
                    f"fpr={det['xgboost']['fpr']:.4f} | attack top1={attack.top1_accuracy:.4f}, "
                    f"top5={attack.top5_accuracy:.4f}")

                sweep.append({
                    "epsilon_total": eps_total,
                    "epsilon_per_feature": eps_per_feature,
                    "d_numeric_features": d_check,
                    "detection_xgboost": det["xgboost"],
                    "privacy_risk": {
                        "top1_accuracy": attack.top1_accuracy,
                        "top5_accuracy": attack.top5_accuracy,
                        "majority_baseline_accuracy": attack.majority_baseline_accuracy,
                    },
                })

            log(f"[{dataset_key}] -- robustness check: all 3 models at eps_total={REPRESENTATIVE_EPSILON_TOTAL} --")
            eps_per_feature_repr = REPRESENTATIVE_EPSILON_TOTAL / d_numeric
            train_noisy, test_noisy, _, _, _ = add_laplace_noise_clipped(
                train_df, test_df, behavioral_cols, eps_per_feature_repr
            )
            det_all = train_and_evaluate_all(train_noisy, test_noisy, behavioral_cols, tax["binary_label_col"])
            for model_name, metrics in det_all.items():
                log(f"[{dataset_key}][eps_total={REPRESENTATIVE_EPSILON_TOTAL}][{model_name}] f1={metrics['f1']:.4f}, "
                    f"auc={metrics['auc_roc']:.4f}, fpr={metrics['fpr']:.4f}")

            all_results[dataset_key] = {
                "d_numeric_features": d_numeric,
                "epsilon_sweep_xgboost": sweep,
                "representative_epsilon_total": REPRESENTATIVE_EPSILON_TOTAL,
                "robustness_check_all_models": det_all,
            }

        with open(OUT_DIR / "scenario4_dp_results.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        log("\nDone. Results saved.")


if __name__ == "__main__":
    main()
