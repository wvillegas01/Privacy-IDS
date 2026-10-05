"""Reemplaza S4 (antes: Laplace sin clipping, eps=1 sin composicion) por el
mecanismo reformalizado (clipping a percentiles + composicion basica,
eps_total=100 repartido entre las d features numericas) en las 5 semillas,
y recalcula Friedman/Wilcoxon con el S4 actualizado.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "03_ataque_reidentificacion" / "01_scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "04_escenario1_baseline" / "01_scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "07_escenario4_dp_ruido" / "01_scripts"))

from attack_module import build_cicids_target, run_attack  # noqa: E402
from detection_module import train_and_evaluate_all  # noqa: E402
from dp_module import add_laplace_noise_clipped  # noqa: E402

STAGE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = STAGE_DIR / "02_outputs"
LOG_DIR = STAGE_DIR / "04_logs"
PREPROC_OUT = STAGE_DIR.parent / "02_preprocesamiento_baseline" / "02_outputs"
SEEDS = [42, 43, 44, 45, 46]
REPRESENTATIVE_EPSILON_TOTAL = 100


def main():
    log_path = LOG_DIR / "03_update_s4_clipped_dp_log.txt"
    with open(log_path, "w", encoding="utf-8") as fh:
        def log(msg):
            print(msg)
            fh.write(msg + "\n")

        with open(OUT_DIR / "cross_scenario_comparison.json") as f:
            existing = json.load(f)

        for dataset_key, taxonomy_file, target_col in [
            ("CICIDS2017", "cicids2017_taxonomy.json", "Source IP"),
            ("UNSW-NB15", "unsw_nb15_taxonomy.json", "srcip"),
        ]:
            with open(PREPROC_OUT / taxonomy_file) as f:
                tax = json.load(f)
            train_df = pd.read_parquet(tax["train_path"])
            test_df = pd.read_parquet(tax["test_path"])
            behavioral_cols = tax["feature_cols"]
            d_numeric = sum(1 for c in behavioral_cols if pd.api.types.is_numeric_dtype(train_df[c]))
            eps_per_feature = REPRESENTATIVE_EPSILON_TOTAL / d_numeric

            if dataset_key == "CICIDS2017":
                target_train, target_test = build_cicids_target(train_df, test_df, top_n=20, id_col=target_col)
            else:
                target_train, target_test = train_df[target_col], test_df[target_col]

            f1_list, risk_list = [], []
            for seed in SEEDS:
                train_noisy, test_noisy, _, _, _ = add_laplace_noise_clipped(
                    train_df, test_df, behavioral_cols, eps_per_feature, random_state=seed
                )
                det = train_and_evaluate_all(train_noisy, test_noisy, behavioral_cols, tax["binary_label_col"],
                                              random_state=seed, model_names=["xgboost"])
                att = run_attack(train_noisy, test_noisy, behavioral_cols, target_train, target_test, random_state=seed)
                f1_list.append(det["xgboost"]["f1"])
                risk_list.append(att.top1_accuracy)
                log(f"[{dataset_key}][seed={seed}] S4(clipped,eps_total=100) F1={det['xgboost']['f1']:.4f}, "
                    f"risk={att.top1_accuracy:.4f}")

            # replace old key
            existing[dataset_key]["f1_by_scenario"].pop("S4_dp_eps1", None)
            existing[dataset_key]["risk_by_scenario"].pop("S4_dp_eps1", None)
            existing[dataset_key]["f1_by_scenario"]["S4_dp_clipped_eps100"] = f1_list
            existing[dataset_key]["risk_by_scenario"]["S4_dp_clipped_eps100"] = risk_list
            log(f"[{dataset_key}] S4 mean F1={np.mean(f1_list):.4f}, mean risk={np.mean(risk_list):.4f}")

            scenario_names = ["S0_behavioral_only", "S1_naive", "S2_anonymized", "S3_reduced", "S4_dp_clipped_eps100"]
            f1_arrays = [np.array(existing[dataset_key]["f1_by_scenario"][s]) for s in scenario_names]
            risk_arrays = [np.array(existing[dataset_key]["risk_by_scenario"][s]) for s in scenario_names]
            friedman_f1 = friedmanchisquare(*f1_arrays)
            friedman_risk = friedmanchisquare(*risk_arrays)
            n, k = 5, 5
            kendall_w_f1 = friedman_f1.statistic / (n * (k - 1))
            kendall_w_risk = friedman_risk.statistic / (n * (k - 1))
            log(f"[{dataset_key}] Friedman(5, S4 updated) F1: stat={friedman_f1.statistic:.4f}, p={friedman_f1.pvalue:.6f}, W={kendall_w_f1:.4f}")
            log(f"[{dataset_key}] Friedman(5, S4 updated) risk: stat={friedman_risk.statistic:.4f}, p={friedman_risk.pvalue:.6f}, W={kendall_w_risk:.4f}")

            bonferroni_alpha = 0.05 / 3
            w_f1 = wilcoxon(existing[dataset_key]["f1_by_scenario"]["S0_behavioral_only"], f1_list)
            w_risk = wilcoxon(existing[dataset_key]["risk_by_scenario"]["S0_behavioral_only"], risk_list)
            log(f"[{dataset_key}] S0_vs_S4(clipped): F1 p={w_f1.pvalue:.6f}, risk p={w_risk.pvalue:.6f}")

            existing[dataset_key]["friedman_f1_5scenarios"] = {"statistic": float(friedman_f1.statistic), "p_value": float(friedman_f1.pvalue)}
            existing[dataset_key]["friedman_risk_5scenarios"] = {"statistic": float(friedman_risk.statistic), "p_value": float(friedman_risk.pvalue)}
            existing[dataset_key]["kendall_w_5scenarios"] = {"f1": kendall_w_f1, "risk": kendall_w_risk}
            existing[dataset_key]["pairwise_clean_axis_isolated"]["S0_vs_S4_behavioral_axis"] = {
                "f1_p": float(w_f1.pvalue), "f1_significant_bonferroni": bool(w_f1.pvalue < bonferroni_alpha),
                "risk_p": float(w_risk.pvalue), "risk_significant_bonferroni": bool(w_risk.pvalue < bonferroni_alpha),
            }
            existing[dataset_key]["s4_epsilon_total"] = REPRESENTATIVE_EPSILON_TOTAL
            existing[dataset_key]["s4_d_numeric_features"] = d_numeric

        with open(OUT_DIR / "cross_scenario_comparison.json", "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)
        log("\nDone. S4 updated to clipped+composed mechanism.")


if __name__ == "__main__":
    main()
