"""Modulo de entrenamiento/evaluacion de deteccion, compartido por las etapas
04 (baseline), 05 (anonimizacion), 06 (reduccion de features) y 07 (DP).

3 modelos por diseno (RF + XGBoost + red neuronal simple) para que el
trade-off privacidad-utilidad no dependa de la eleccion de un unico
algoritmo. Clasificacion binaria (ataque vs. benigno/normal) sobre
binary_label_col de cada taxonomia -- es el framing estandar de la
literatura de IDS para F1/AUC-ROC/FPR.

Mismo cap de submuestreo estratificado de entrenamiento (300,000 filas) que
el modulo de ataque de la etapa 03, para que la descripcion metodologica
sea unica y consistente entre deteccion y ataque de re-identificacion. El
test set se evalua completo, sin submuestrear.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "03_ataque_reidentificacion" / "01_scripts"))
from attack_module import encode_features  # noqa: E402

RANDOM_SEED = 42
MAX_DETECTION_TRAIN_ROWS = 300_000


def make_models(random_state: int = RANDOM_SEED):
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=20, n_jobs=-1, random_state=random_state,
            class_weight="balanced_subsample",
        ),
        "xgboost": XGBClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.1, n_jobs=-1,
            random_state=random_state, eval_metric="logloss",
            tree_method="hist",
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=100, random_state=random_state,
            early_stopping=True, n_iter_no_change=5,
        ),
    }


def stratified_subsample(X, y, cap, random_state=RANDOM_SEED):
    n = len(y)
    if n <= cap:
        return X, y
    rng = np.random.RandomState(random_state)
    idx = rng.choice(n, size=cap, replace=False)
    return X[idx], y[idx]


def compute_metrics(y_true, y_pred, y_proba):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    multiclass_ok = len(np.unique(y_true)) > 1
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_proba)) if multiclass_ok else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_proba)) if multiclass_ok else float("nan"),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "fpr": float(fpr),
        "n_test": int(len(y_true)),
    }
    if multiclass_ok:
        prec_curve, rec_curve, thr_curve = precision_recall_curve(y_true, y_proba)
        f1_curve = np.where((prec_curve + rec_curve) > 0,
                             2 * prec_curve * rec_curve / (prec_curve + rec_curve + 1e-12), 0.0)
        best_idx = int(np.argmax(f1_curve[:-1])) if len(f1_curve) > 1 else 0
        metrics["best_f1_test_optimal_threshold"] = float(f1_curve[best_idx])
        metrics["best_threshold_test_optimal"] = float(thr_curve[best_idx]) if len(thr_curve) else 0.5
    else:
        metrics["best_f1_test_optimal_threshold"] = float("nan")
        metrics["best_threshold_test_optimal"] = float("nan")
    return metrics


def train_and_evaluate_all(train_df, test_df, feature_cols, label_col, random_state=RANDOM_SEED, model_names=None):
    """Encodes categorical features once, trains RF+XGBoost+MLP (or a subset
    via model_names, e.g. ["xgboost"] for a fast sweep) on a shared
    stratified subsample, evaluates on the full test split.
    Returns {model_name: metrics_dict}.
    """
    train_enc, test_enc = encode_features(train_df, test_df, feature_cols)

    X_train_full = train_enc.to_numpy(dtype="float64", na_value=0.0)
    y_train_full = train_df[label_col].to_numpy()
    X_train, y_train = stratified_subsample(X_train_full, y_train_full, MAX_DETECTION_TRAIN_ROWS, random_state)

    X_test = test_enc.to_numpy(dtype="float64", na_value=0.0)
    y_test = test_df[label_col].to_numpy()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}
    all_models = make_models(random_state)
    models = {k: v for k, v in all_models.items() if model_names is None or k in model_names}
    for name, model in models.items():
        if name == "mlp":
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
        results[name] = compute_metrics(y_test, y_pred, y_proba)
        results[name]["n_train_used"] = int(len(y_train))

    return results
