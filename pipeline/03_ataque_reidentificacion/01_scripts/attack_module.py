"""Modulo de ataque de re-identificacion / linkage, compartido por las etapas
03 (linea base + ranking de sensibilidad) y 05/06/07 (escenarios 2/3/4).

Amenaza modelada: un atacante con conocimiento de fondo (el split de train,
con features + identidad real) intenta inferir la identidad real (host) de
un registro publicado (el split de test) usando SOLO las features de
comportamiento -- nunca las columnas identificador, que son justamente lo
que cada escenario de privacidad transforma o elimina.

Metrica de riesgo = accuracy top-1 (y top-5) del atacante sobre el test set,
comparada contra la accuracy de un clasificador de mayoria (chance level).
Un valor de accuracy muy por encima del chance level indica que la
identidad sigue siendo inferible desde el comportamiento (fuga residual),
incluso si los identificadores directos fueron removidos/transformados.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

RANDOM_SEED = 42
MAX_ATTACK_TRAIN_ROWS = 300_000  # subsample cap for tractability, stratified on target


@dataclass
class AttackResult:
    top1_accuracy: float
    top5_accuracy: float
    majority_baseline_accuracy: float
    n_classes: int
    n_train_used: int
    n_test: int
    feature_importances: dict = field(default_factory=dict)


def build_cicids_target(train_df: pd.DataFrame, test_df: pd.DataFrame, top_n: int = 20, id_col: str = "Source IP"):
    """Bucket the long tail of external IPs into 'Other' -- CICIDS2017 has
    ~16k distinct Source IP values but the top 20 (firewall + 12 victim
    hosts + a couple of infra addresses) cover ~88% of traffic. Modeling
    16k raw classes would be intractable and would mostly measure noise in
    the tail rather than genuine host re-identification.
    """
    top_ips = train_df[id_col].value_counts().head(top_n).index
    train_target = train_df[id_col].where(train_df[id_col].isin(top_ips), "Other")
    test_target = test_df[id_col].where(test_df[id_col].isin(top_ips), "Other")
    return train_target, test_target


def encode_features(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list):
    """Label-encode nominal/object feature columns (e.g. proto/service/state
    in UNSW-NB15) so every model here gets a purely numeric matrix. Unseen
    test-time categories map to a dedicated 'unknown' code rather than
    crashing. Returns two float64 DataFrames restricted to feature_cols.
    """
    train_enc = train_df[feature_cols].copy()
    test_enc = test_df[feature_cols].copy()
    for c in feature_cols:
        if train_enc[c].dtype == object or str(train_enc[c].dtype) == "category":
            categories = pd.Index(train_enc[c].astype(str).unique())
            code_map = {v: i for i, v in enumerate(categories)}
            unknown_code = len(categories)
            train_enc[c] = train_enc[c].astype(str).map(code_map).astype("float64")
            test_enc[c] = test_enc[c].astype(str).map(code_map).fillna(unknown_code).astype("float64")
    return train_enc, test_enc


def run_attack(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list,
    target_train: pd.Series,
    target_test: pd.Series,
    random_state: int = RANDOM_SEED,
) -> AttackResult:
    train_enc, test_enc = encode_features(train_df, test_df, feature_cols)
    X_train_full = train_enc.to_numpy(dtype="float64", na_value=0.0)
    y_train_full = target_train.to_numpy()

    n_train_full = len(y_train_full)
    if n_train_full > MAX_ATTACK_TRAIN_ROWS:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(n_train_full, size=MAX_ATTACK_TRAIN_ROWS, replace=False)
        X_train = X_train_full[idx]
        y_train = y_train_full[idx]
    else:
        X_train, y_train = X_train_full, y_train_full

    X_test = test_enc.to_numpy(dtype="float64", na_value=0.0)
    y_test = target_test.to_numpy()

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        n_jobs=-1,
        random_state=random_state,
        class_weight="balanced_subsample",
    )
    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)
    classes = clf.classes_
    top1_pred = classes[np.argmax(proba, axis=1)]
    top1_acc = float(np.mean(top1_pred == y_test))

    k = min(5, len(classes))
    top5_idx = np.argsort(-proba, axis=1)[:, :k]
    top5_pred = classes[top5_idx]
    top5_acc = float(np.mean([(y_test[i] in top5_pred[i]) for i in range(len(y_test))]))

    majority_class = pd.Series(y_train).value_counts().idxmax()
    majority_acc = float(np.mean(y_test == majority_class))

    importances = dict(zip(feature_cols, clf.feature_importances_.tolist()))

    return AttackResult(
        top1_accuracy=top1_acc,
        top5_accuracy=top5_acc,
        majority_baseline_accuracy=majority_acc,
        n_classes=len(classes),
        n_train_used=len(y_train),
        n_test=len(y_test),
        feature_importances=importances,
    )
