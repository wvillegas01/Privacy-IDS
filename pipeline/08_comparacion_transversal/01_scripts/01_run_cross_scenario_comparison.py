"""Etapa 08 - Comparacion transversal de las 4 tecnicas de privacidad.

Los escenarios 04-07 dieron UN punto por tecnica (seed=42). Para poder
correr tests estadisticos pareados (Friedman + Wilcoxon con correccion de
Bonferroni) hace falta una distribucion, no un punto -- se repiten aqui las
4 tecnicas con 5 semillas distintas (XGBoost unicamente, el modelo mas
fuerte y consistente en todas las etapas previas; no los 3 modelos, por
costo computacional) y se comparan F1 de deteccion y top-1 del ataque de
re-identificacion entre escenarios.

Escenario 4 (DP) se fija en eps=1 para esta comparacion de 4 vias (punto
medio del barrido de la etapa 07); la curva completa de epsilon sigue
siendo la evidencia primaria de ese escenario, esto es solo para hacerlo
comparable a los otros 3 en una misma tabla.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "03_ataque_reidentificacion" / "01_scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "04_escenario1_baseline" / "01_scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "05_escenario2_anonimizacion" / "01_scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "07_escenario4_dp_ruido" / "01_scripts"))

from attack_module import build_cicids_target, run_attack  # noqa: E402
from detection_module import train_and_evaluate_all  # noqa: E402
from anonymization_module import CICIDS_RAW_ID_FEATURES, UNSW_RAW_ID_FEATURES, generalize_ip, generalize_port  # noqa: E402
from dp_module import add_laplace_noise  # noqa: E402

STAGE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = STAGE_DIR / "02_outputs"
LOG_DIR = STAGE_DIR / "04_logs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

PREPROC_OUT = STAGE_DIR.parent / "02_preprocesamiento_baseline" / "02_outputs"
STAGE06_OUT = STAGE_DIR.parent / "06_escenario3_reduccion_features" / "02_outputs"

SEEDS = [42, 43, 44, 45, 46]
DP_EPSILON_FOR_COMPARISON = 1.0


def anonymize_columns(df, dataset):
    df = df.copy()
    if dataset == "CICIDS2017":
        ip_cols, port_cols = ["Source IP", "Destination IP"], ["Source Port", "Destination Port"]
    else:
        ip_cols, port_cols = ["srcip", "dstip"], ["sport", "dsport"]
    anon_cols = []
    for c in ip_cols:
        df[f"{c}_anon"] = generalize_ip(df[c])
        anon_cols.append(f"{c}_anon")
    for c in port_cols:
        df[f"{c}_anon"] = generalize_port(df[c])
        anon_cols.append(f"{c}_anon")
    return df, anon_cols


def main():
    log_path = LOG_DIR / "01_run_cross_scenario_comparison_log.txt"
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
            with open(STAGE06_OUT / "scenario3_feature_reduction_results.json") as f:
                s3 = json.load(f)

            train_df = pd.read_parquet(tax["train_path"])
            test_df = pd.read_parquet(tax["test_path"])
            behavioral_cols = tax["feature_cols"]
            binary_label_col = tax["binary_label_col"]

            train_anon, anon_cols = anonymize_columns(train_df, dataset_key)
            test_anon, _ = anonymize_columns(test_df, dataset_key)

            removed_feats = s3[dataset_key]["removed_features"]
            reduced_cols = [c for c in behavioral_cols if c not in removed_feats]

            if dataset_key == "CICIDS2017":
                target_train, target_test = build_cicids_target(train_df, test_df, top_n=20, id_col=target_col)
                target_train_anon, target_test_anon = target_train, target_test
            else:
                target_train, target_test = train_df[target_col], test_df[target_col]
                target_train_anon, target_test_anon = target_train, target_test

            f1_by_scenario = {"S1_naive": [], "S2_anonymized": [], "S3_reduced": [], "S4_dp_eps1": []}
            risk_by_scenario = {"S1_naive": [], "S2_anonymized": [], "S3_reduced": [], "S4_dp_eps1": []}

            for seed in SEEDS:
                naive_cols = behavioral_cols + raw_id_cols
                det = train_and_evaluate_all(train_df, test_df, naive_cols, binary_label_col, random_state=seed, model_names=["xgboost"])
                evidence = [c for c in naive_cols if c != target_col]
                att = run_attack(train_df, test_df, evidence, target_train, target_test, random_state=seed)
                f1_by_scenario["S1_naive"].append(det["xgboost"]["f1"])
                risk_by_scenario["S1_naive"].append(att.top1_accuracy)

                anon_feat_cols = behavioral_cols + anon_cols
                det = train_and_evaluate_all(train_anon, test_anon, anon_feat_cols, binary_label_col, random_state=seed, model_names=["xgboost"])
                att = run_attack(train_anon, test_anon, anon_feat_cols, target_train_anon, target_test_anon, random_state=seed)
                f1_by_scenario["S2_anonymized"].append(det["xgboost"]["f1"])
                risk_by_scenario["S2_anonymized"].append(att.top1_accuracy)

                det = train_and_evaluate_all(train_df, test_df, reduced_cols, binary_label_col, random_state=seed, model_names=["xgboost"])
                att = run_attack(train_df, test_df, reduced_cols, target_train, target_test, random_state=seed)
                f1_by_scenario["S3_reduced"].append(det["xgboost"]["f1"])
                risk_by_scenario["S3_reduced"].append(att.top1_accuracy)

                train_noisy, test_noisy, _ = add_laplace_noise(train_df, test_df, behavioral_cols, DP_EPSILON_FOR_COMPARISON, random_state=seed)
                det = train_and_evaluate_all(train_noisy, test_noisy, behavioral_cols, binary_label_col, random_state=seed, model_names=["xgboost"])
                att = run_attack(train_noisy, test_noisy, behavioral_cols, target_train, target_test, random_state=seed)
                f1_by_scenario["S4_dp_eps1"].append(det["xgboost"]["f1"])
                risk_by_scenario["S4_dp_eps1"].append(att.top1_accuracy)

                log(f"[{dataset_key}][seed={seed}] F1: " + ", ".join(f"{k}={v[-1]:.4f}" for k, v in f1_by_scenario.items()))
                log(f"[{dataset_key}][seed={seed}] risk: " + ", ".join(f"{k}={v[-1]:.4f}" for k, v in risk_by_scenario.items()))

            scenario_names = list(f1_by_scenario.keys())
            f1_arrays = [np.array(f1_by_scenario[s]) for s in scenario_names]
            risk_arrays = [np.array(risk_by_scenario[s]) for s in scenario_names]

            friedman_f1 = friedmanchisquare(*f1_arrays)
            friedman_risk = friedmanchisquare(*risk_arrays)
            log(f"[{dataset_key}] Friedman F1 across 4 scenarios: stat={friedman_f1.statistic:.4f}, p={friedman_f1.pvalue:.6f}")
            log(f"[{dataset_key}] Friedman risk across 4 scenarios: stat={friedman_risk.statistic:.4f}, p={friedman_risk.pvalue:.6f}")

            pairwise = {}
            n_pairs = 3  # S1 vs S2, S1 vs S3, S1 vs S4
            bonferroni_alpha = 0.05 / n_pairs
            for other in ["S2_anonymized", "S3_reduced", "S4_dp_eps1"]:
                w_f1 = wilcoxon(f1_by_scenario["S1_naive"], f1_by_scenario[other])
                w_risk = wilcoxon(risk_by_scenario["S1_naive"], risk_by_scenario[other])
                pairwise[f"S1_vs_{other}"] = {
                    "f1_wilcoxon_stat": float(w_f1.statistic), "f1_p": float(w_f1.pvalue),
                    "f1_significant_bonferroni": bool(w_f1.pvalue < bonferroni_alpha),
                    "risk_wilcoxon_stat": float(w_risk.statistic), "risk_p": float(w_risk.pvalue),
                    "risk_significant_bonferroni": bool(w_risk.pvalue < bonferroni_alpha),
                }
                log(f"[{dataset_key}] S1 vs {other}: F1 p={w_f1.pvalue:.6f} (Bonferroni alpha={bonferroni_alpha:.4f}), "
                    f"risk p={w_risk.pvalue:.6f}")

            all_results[dataset_key] = {
                "seeds": SEEDS,
                "f1_by_scenario": {k: list(v) for k, v in f1_by_scenario.items()},
                "risk_by_scenario": {k: list(v) for k, v in risk_by_scenario.items()},
                "friedman_f1": {"statistic": float(friedman_f1.statistic), "p_value": float(friedman_f1.pvalue)},
                "friedman_risk": {"statistic": float(friedman_risk.statistic), "p_value": float(friedman_risk.pvalue)},
                "pairwise_wilcoxon_vs_S1": pairwise,
            }

        with open(OUT_DIR / "cross_scenario_comparison.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        log("\nDone. Results saved.")


if __name__ == "__main__":
    main()
