"""Etapa 08 (extension) - Añade S0 (comportamiento puro, sin identificador
alguno) al mismo barrido de 5 semillas que S1-S4, corrigiendo una confusion
de la version 1 del manuscrito: S3 y S4 ya se computaban sobre features de
comportamiento sin identificadores (correcto en el codigo desde el inicio),
pero el texto los comparaba contra S1 (que SI tiene identificadores crudos)
en vez de contra S0 -- una comparacion no aislada, senalada correctamente en
la revision del usuario. S0 es la comparacion limpia para S3 y S4; S1 sigue
siendo la comparacion limpia para S2 (ambos con comportamiento completo,
difieren solo en el tratamiento del identificador).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "03_ataque_reidentificacion" / "01_scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "04_escenario1_baseline" / "01_scripts"))

from attack_module import build_cicids_target, run_attack  # noqa: E402
from detection_module import train_and_evaluate_all  # noqa: E402

STAGE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = STAGE_DIR / "02_outputs"
LOG_DIR = STAGE_DIR / "04_logs"

PREPROC_OUT = STAGE_DIR.parent / "02_preprocesamiento_baseline" / "02_outputs"
SEEDS = [42, 43, 44, 45, 46]


def main():
    log_path = LOG_DIR / "02_add_s0_scenario_log.txt"
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

            if dataset_key == "CICIDS2017":
                target_train, target_test = build_cicids_target(train_df, test_df, top_n=20, id_col=target_col)
            else:
                target_train, target_test = train_df[target_col], test_df[target_col]

            f1_list, risk_list = [], []
            for seed in SEEDS:
                det = train_and_evaluate_all(
                    train_df, test_df, behavioral_cols, tax["binary_label_col"],
                    random_state=seed, model_names=["xgboost"],
                )
                att = run_attack(train_df, test_df, behavioral_cols, target_train, target_test, random_state=seed)
                f1_list.append(det["xgboost"]["f1"])
                risk_list.append(att.top1_accuracy)
                log(f"[{dataset_key}][seed={seed}] S0 F1={det['xgboost']['f1']:.4f}, "
                    f"risk={att.top1_accuracy:.4f}")

            existing[dataset_key]["f1_by_scenario"]["S0_behavioral_only"] = f1_list
            existing[dataset_key]["risk_by_scenario"]["S0_behavioral_only"] = risk_list
            log(f"[{dataset_key}] S0 mean F1={np.mean(f1_list):.4f} (sd={np.std(f1_list):.4f}), "
                f"mean risk={np.mean(risk_list):.4f} (sd={np.std(risk_list):.4f})")

            # Recompute Friedman/Wilcoxon over the 5-scenario set (S0-S4)
            scenario_names = ["S0_behavioral_only", "S1_naive", "S2_anonymized", "S3_reduced", "S4_dp_eps1"]
            f1_arrays = [np.array(existing[dataset_key]["f1_by_scenario"][s]) for s in scenario_names]
            risk_arrays = [np.array(existing[dataset_key]["risk_by_scenario"][s]) for s in scenario_names]
            friedman_f1 = friedmanchisquare(*f1_arrays)
            friedman_risk = friedmanchisquare(*risk_arrays)
            log(f"[{dataset_key}] Friedman(5 scenarios) F1: stat={friedman_f1.statistic:.4f}, p={friedman_f1.pvalue:.6f}")
            log(f"[{dataset_key}] Friedman(5 scenarios) risk: stat={friedman_risk.statistic:.4f}, p={friedman_risk.pvalue:.6f}")
            n, k = 5, 5
            kendall_w_f1 = friedman_f1.statistic / (n * (k - 1))
            kendall_w_risk = friedman_risk.statistic / (n * (k - 1))
            log(f"[{dataset_key}] Kendall's W (5 scenarios): F1={kendall_w_f1:.4f}, risk={kendall_w_risk:.4f}")

            # Clean pairwise comparisons: S1 vs S2 (identifier axis), S0 vs S3, S0 vs S4 (behavioral axis)
            bonferroni_alpha = 0.05 / 3
            pairwise_clean = {}
            for label, a, b in [
                ("S1_vs_S2_identifier_axis", "S1_naive", "S2_anonymized"),
                ("S0_vs_S3_behavioral_axis", "S0_behavioral_only", "S3_reduced"),
                ("S0_vs_S4_behavioral_axis", "S0_behavioral_only", "S4_dp_eps1"),
            ]:
                w_f1 = wilcoxon(existing[dataset_key]["f1_by_scenario"][a], existing[dataset_key]["f1_by_scenario"][b])
                w_risk = wilcoxon(existing[dataset_key]["risk_by_scenario"][a], existing[dataset_key]["risk_by_scenario"][b])
                pairwise_clean[label] = {
                    "f1_p": float(w_f1.pvalue), "f1_significant_bonferroni": bool(w_f1.pvalue < bonferroni_alpha),
                    "risk_p": float(w_risk.pvalue), "risk_significant_bonferroni": bool(w_risk.pvalue < bonferroni_alpha),
                }
                log(f"[{dataset_key}] {label}: F1 p={w_f1.pvalue:.6f}, risk p={w_risk.pvalue:.6f}")

            existing[dataset_key]["friedman_f1_5scenarios"] = {"statistic": float(friedman_f1.statistic), "p_value": float(friedman_f1.pvalue)}
            existing[dataset_key]["friedman_risk_5scenarios"] = {"statistic": float(friedman_risk.statistic), "p_value": float(friedman_risk.pvalue)}
            existing[dataset_key]["kendall_w_5scenarios"] = {"f1": kendall_w_f1, "risk": kendall_w_risk}
            existing[dataset_key]["pairwise_clean_axis_isolated"] = pairwise_clean

        with open(OUT_DIR / "cross_scenario_comparison.json", "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)
        log("\nDone. Updated cross_scenario_comparison.json with S0.")


if __name__ == "__main__":
    main()
