"""Modulo de privacidad diferencial a NIVEL DE DATO/INPUT (mecanismo de
Laplace sobre features numericas), para la etapa 07 (Escenario 4).

Diferenciador clave frente al portafolio (ver ledger de la etapa 00): esto
es ruido sobre los VALORES DE ENTRADA antes de entrenar un modelo
centralizado, no DP-SGD sobre gradientes en un sistema federado (que es lo
que hace el paper 'Seguridad y privacidad' en revision).

v2 (2026-09-16), tras revision del usuario: la v1 aproximaba la sensibilidad
por el rango observado (max-min), sin recorte (clipping) previo -- senalado
correctamente como demasiado conservador (dominado por outliers) y sin una
cota de dominio fijada a priori, sin definir adyacencia, y sin contabilizar
la composicion del presupuesto al aplicar el mecanismo a muchas features a
la vez. Esta version:

  1. Recorta (clip) cada feature numerica al rango [P1, P99] calculado
     SOLO en train antes de anadir ruido -- una cota de sensibilidad fijada
     por el mecanismo mismo (independiente del dato de una fila concreta),
     no simplemente el rango observado. Sigue siendo empirica (se estima de
     los datos, no de una especificacion de dominio externa), pero es la
     relajacion practica estandar en la literatura de DP aplicada a datos
     tabulares cuando no existe una cota de dominio publicada.
  2. Adyacencia declarada explicitamente: dos conjuntos de entrenamiento son
     vecinos si difieren en como maximo un registro (adyacencia a nivel de
     registro, la definicion estandar de DP centralizada).
  3. Composicion: el mecanismo se aplica de forma independiente a d features
     numericas, cada una con presupuesto epsilon_feature. Por el teorema de
     composicion basica (Dwork y Roth, 2014), la publicacion conjunta de
     las d features satisface (d * epsilon_feature)-DP a nivel de registro
     completo. epsilon_total = d * epsilon_feature se calcula y reporta
     explicitamente junto con epsilon_feature en todos los resultados.

Las features categoricas (proto/service/state en UNSW-NB15) no reciben
ruido -- el mecanismo de Laplace no aplica a variables nominales; se
documenta como alcance, no como omision oculta.
"""
import numpy as np
import pandas as pd

CLIP_LOWER_PERCENTILE = 1.0
CLIP_UPPER_PERCENTILE = 99.0


def add_laplace_noise_clipped(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list,
                               epsilon_per_feature: float, random_state: int = 42):
    """Ruido de Laplace con sensibilidad por recorte a percentiles [1,99]
    calculados solo en train. Devuelve train/test perturbados, las
    sensibilidades por columna, el numero de features numericas
    perturbadas (d) y epsilon_total = d * epsilon_per_feature (composicion
    basica, adyacencia a nivel de registro).
    """
    rng = np.random.RandomState(random_state)
    train_noisy = train_df.copy()
    test_noisy = test_df.copy()

    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train_df[c])]
    sensitivities = {}
    for c in numeric_cols:
        col = train_df[c].to_numpy(dtype="float64")
        col = col[np.isfinite(col)]
        if len(col) == 0:
            sensitivities[c] = 0.0
            continue
        lo, hi = np.percentile(col, [CLIP_LOWER_PERCENTILE, CLIP_UPPER_PERCENTILE])
        sensitivity = float(hi - lo)
        sensitivities[c] = sensitivity

        train_col = train_df[c].to_numpy(dtype="float64")
        test_col = test_df[c].to_numpy(dtype="float64")
        train_col_clipped = np.clip(train_col, lo, hi)
        test_col_clipped = np.clip(test_col, lo, hi)

        scale = sensitivity / epsilon_per_feature if sensitivity > 0 and epsilon_per_feature > 0 else 0.0
        if scale > 0:
            train_noisy[c] = train_col_clipped + rng.laplace(0, scale, size=len(train_df))
            test_noisy[c] = test_col_clipped + rng.laplace(0, scale, size=len(test_df))
        else:
            train_noisy[c] = train_col_clipped
            test_noisy[c] = test_col_clipped

    d = len(numeric_cols)
    epsilon_total = d * epsilon_per_feature
    return train_noisy, test_noisy, sensitivities, d, epsilon_total


def add_laplace_noise(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list, epsilon: float, random_state: int = 42):
    """v1, conservada solo como referencia de comparacion (rango completo,
    sin clipping) -- ya no es el mecanismo primario del manuscrito.
    """
    rng = np.random.RandomState(random_state)
    train_noisy = train_df.copy()
    test_noisy = test_df.copy()

    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train_df[c])]
    sensitivities = {}
    for c in numeric_cols:
        col = train_df[c].to_numpy(dtype="float64")
        col = col[np.isfinite(col)]
        sensitivity = float(np.max(col) - np.min(col)) if len(col) else 0.0
        sensitivities[c] = sensitivity
        scale = sensitivity / epsilon if sensitivity > 0 and epsilon > 0 else 0.0
        if scale > 0:
            train_noisy[c] = train_df[c].to_numpy(dtype="float64") + rng.laplace(0, scale, size=len(train_df))
            test_noisy[c] = test_df[c].to_numpy(dtype="float64") + rng.laplace(0, scale, size=len(test_df))

    return train_noisy, test_noisy, sensitivities
