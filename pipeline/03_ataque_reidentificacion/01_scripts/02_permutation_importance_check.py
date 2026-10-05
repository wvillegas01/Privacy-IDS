"""Verifica que el ranking de importancia por impureza (usado para definir
R, el criterio de seleccion de S3) es consistente con importancia por
permutacion, pedido por el usuario porque la importancia por impureza de
Random Forest puede favorecer variables continuas/de alta cardinalidad.
Solo se recalcula para las top-15 features de cada dataset, sobre una
submuestra del test set, por costo computacional.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

sys.path.insert(0, str(Path(__file__).resolve().parent))
from attack_module import build_cicids_target, encode_features, MAX_ATTACK_TRAIN_ROWS  # noqa: E402

STAGE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = STAGE_DIR / "02_outputs"
LOG_DIR = STAGE_DIR / "04_logs"
PREPROC_OUT = STAGE_DIR.parent / "02_preprocesamiento_baseline" / "02_outputs"

N_TOP_FEATURES = 15
N_PERM_TEST_SAMPLE = 30_000


def main():
    log_path = LOG_DIR / "02_permutation_importance_check_log.txt"
    with open(log_path, "w", encoding="utf-8") as fh:
        def log(msg):
            print(msg)
            fh.write(msg + "\n")

        with open(OUT_DIR / "baseline_attack_results.json") as f:
            attack_results = json.load(f)

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

            train_enc, test_enc = encode_features(train_df, test_df, behavioral_cols)
            X_train_full = train_enc.to_numpy(dtype="float64", na_value=0.0)
            y_train_full = target_train.to_numpy()
            rng = np.random.RandomState(42)
            idx = rng.choice(len(y_train_full), size=min(MAX_ATTACK_TRAIN_ROWS, len(y_train_full)), replace=False)
            X_train, y_train = X_train_full[idx], y_train_full[idx]

            clf = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42,
                                          class_weight="balanced_subsample")
            clf.fit(X_train, y_train)

            X_test_full = test_enc.to_numpy(dtype="float64", na_value=0.0)
            y_test_full = target_test.to_numpy()
            n_sample = min(N_PERM_TEST_SAMPLE, len(y_test_full))
            test_idx = rng.choice(len(y_test_full), size=n_sample, replace=False)
            X_test_sample = X_test_full[test_idx]
            y_test_sample = y_test_full[test_idx]

            impurity_ranking = attack_results[dataset_key]["feature_importance_ranking"]
            top_features = [f for f, _ in impurity_ranking[:N_TOP_FEATURES]]
            top_idx = [behavioral_cols.index(f) for f in top_features]

            # permutation_importance must see the FULL feature matrix (the
            # classifier expects all columns); we only shuffle/report the
            # top-N columns to keep the computation bounded.
            perm = permutation_importance(clf, X_test_sample, y_test_sample,
                                           n_repeats=5, random_state=42, n_jobs=-1,
                                           max_samples=1.0)
            perm_full = dict(zip(behavioral_cols, perm.importances_mean))
            perm_ranking = sorted(((f, perm_full[f]) for f in top_features), key=lambda kv: -kv[1])

            impurity_rank_pos = {f: i for i, f in enumerate(top_features)}
            perm_rank_pos = {f: i for i, (f, _) in enumerate(perm_ranking)}
            rho, pval = spearmanr([impurity_rank_pos[f] for f in top_features],
                                   [perm_rank_pos[f] for f in top_features])

            log(f"[{dataset_key}] impurity-based top-{N_TOP_FEATURES}: {top_features}")
            log(f"[{dataset_key}] permutation-based ranking (same {N_TOP_FEATURES}): {perm_ranking}")
            log(f"[{dataset_key}] Spearman rank correlation (impurity vs permutation, top-{N_TOP_FEATURES}): "
                f"rho={rho:.4f}, p={pval:.6f}")
            log(f"[{dataset_key}] top-3 by permutation importance: {[f for f,_ in perm_ranking[:3]]}")
            log(f"[{dataset_key}] top-3 by impurity importance: {top_features[:3]}")


if __name__ == "__main__":
    main()
