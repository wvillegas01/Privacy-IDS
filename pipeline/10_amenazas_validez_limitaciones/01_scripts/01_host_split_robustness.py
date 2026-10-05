"""Verificacion de robustez pedida por el usuario (punto #8 de su revision):
la particion 80/20 primaria es aleatoria por fila, lo que permite que flujos
del mismo host aparezcan tanto en train como en test. Esta etapa repite las
condiciones centrales (S0, S1, S3) con una particion por GRUPO DE HOST: cada
direccion IP de origen queda enteramente en train o enteramente en test,
nunca en ambos. Es una verificacion de robustez, no un reemplazo del
analisis primario (que se conserva integro).

Alcance deliberadamente acotado: XGBoost unicamente, una semilla, las 3
condiciones mas relevantes para el hallazgo central (S0 referencia, S1
identificadores crudos, S3 reduccion de features), en vez del barrido
completo de 5 condiciones x 3 modelos x 5 semillas del analisis primario.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "03_ataque_reidentificacion" / "01_scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "04_escenario1_baseline" / "01_scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "05_escenario2_anonimizacion" / "01_scripts"))

from attack_module import build_cicids_target, run_attack  # noqa: E402
from detection_module import train_and_evaluate_all  # noqa: E402
from anonymization_module import CICIDS_RAW_ID_FEATURES, UNSW_RAW_ID_FEATURES  # noqa: E402

STAGE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = STAGE_DIR / "02_outputs"
LOG_DIR = STAGE_DIR / "04_logs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

PREPROC_OUT = STAGE_DIR.parent / "02_preprocesamiento_baseline" / "02_outputs"
STAGE06_OUT = STAGE_DIR.parent / "06_escenario3_reduccion_features" / "02_outputs"
SEED = 42
TEST_FRACTION_HOSTS = 0.2


def group_split(df: pd.DataFrame, group_col: str, test_frac: float, seed: int):
    groups = df[group_col].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(groups)
    n_test = max(1, int(len(groups) * test_frac))
    test_groups = set(groups[:n_test])
    is_test = df[group_col].isin(test_groups)
    return df[~is_test].copy(), df[is_test].copy(), len(test_groups), len(groups)


def main():
    log_path = LOG_DIR / "01_host_split_robustness_log.txt"
    with open(log_path, "w", encoding="utf-8") as fh:
        def log(msg):
            print(msg)
            fh.write(msg + "\n")

        with open(STAGE06_OUT / "scenario3_feature_reduction_results.json") as f:
            s3 = json.load(f)

        results = {}
        for dataset_key, taxonomy_file, raw_id_cols, target_col, group_col in [
            ("CICIDS2017", "cicids2017_taxonomy.json", CICIDS_RAW_ID_FEATURES, "Source IP", "Source IP"),
            ("UNSW-NB15", "unsw_nb15_taxonomy.json", UNSW_RAW_ID_FEATURES, "srcip", "srcip"),
        ]:
            with open(PREPROC_OUT / taxonomy_file) as f:
                tax = json.load(f)
            # Recombine the original train+test (row-split) into one pool, then re-split by host.
            full_df = pd.concat([
                pd.read_parquet(tax["train_path"]),
                pd.read_parquet(tax["test_path"]),
            ], ignore_index=True)

            train_df, test_df, n_test_hosts, n_hosts = group_split(full_df, group_col, TEST_FRACTION_HOSTS, SEED)
            log(f"[{dataset_key}] host-based split: {n_test_hosts}/{n_hosts} hosts held out for test, "
                f"train={len(train_df)} rows, test={len(test_df)} rows")

            behavioral_cols = tax["feature_cols"]
            naive_cols = behavioral_cols + raw_id_cols
            removed_feats = s3[dataset_key]["removed_features"]
            reduced_cols = [c for c in behavioral_cols if c not in removed_feats]

            if dataset_key == "CICIDS2017":
                target_train, target_test = build_cicids_target(train_df, test_df, top_n=20, id_col=target_col)
            else:
                target_train, target_test = train_df[target_col], test_df[target_col]

            dataset_results = {"n_test_hosts": n_test_hosts, "n_hosts": n_hosts,
                                "n_train_rows": len(train_df), "n_test_rows": len(test_df)}

            for cond_name, cond_cols in [("S0_behavioral_only", behavioral_cols),
                                          ("S1_naive", naive_cols),
                                          ("S3_reduced", reduced_cols)]:
                det = train_and_evaluate_all(train_df, test_df, cond_cols, tax["binary_label_col"],
                                              random_state=SEED, model_names=["xgboost"])
                m = det["xgboost"]
                log(f"[{dataset_key}][{cond_name}] F1={m['f1']:.4f}, AUC={m['auc_roc']:.4f}, "
                    f"PR-AUC={m['pr_auc']:.4f}, precision={m['precision']:.4f}, recall={m['recall']:.4f}, "
                    f"specificity={m['specificity']:.4f}, MCC={m['mcc']:.4f}, "
                    f"best_F1(test-optimal thr)={m['best_f1_test_optimal_threshold']:.4f} "
                    f"@thr={m['best_threshold_test_optimal']:.4f}")
                # Re-identification attack metrics on held-out hosts are NOT reported here:
                # a closed-set classifier cannot meaningfully predict an identity class it
                # never saw in training, so top-1/majority-baseline under this split are not
                # well-defined re-identification measurements (see report).
                dataset_results[cond_name] = m

            results[dataset_key] = dataset_results

        with open(OUT_DIR / "host_split_robustness.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        log("\nDone.")


if __name__ == "__main__":
    main()
