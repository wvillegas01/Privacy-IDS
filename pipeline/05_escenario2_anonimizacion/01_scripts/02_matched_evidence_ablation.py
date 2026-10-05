"""Verificacion pedida por el usuario (revision v6->v7): la comparacion S1
vs S2 no aisla limpiamente el efecto de "generalizar en vez de mantener el
identificador exacto", porque S2 le da al atacante un identificador de
origen generalizado (el bloque /16) que S1 no tiene en ninguna forma -- S1
excluye la IP de origen por completo (es la columna objetivo), mientras que
S2 incluye su version generalizada (columna distinta, no excluida).

Esta etapa aisla los dos efectos por separado, en vez de solo declarar la
limitacion en prosa:

  1. S2_sin_bloque_origen: evidencia = comportamiento + IP destino
     generalizada + puertos generalizados (SIN el bloque de origen). Esto
     es evidencia estructuralmente equivalente a S1 (mismos campos
     presentes/ausentes; la unica diferencia es generalizado vs. crudo en
     destino/puertos), por lo que S1 vs. S2_sin_bloque_origen SI aisla
     limpiamente el efecto de generalizar destino y puertos.
  2. S2_completo: evidencia = comportamiento + los 4 identificadores
     generalizados (incluye el bloque de origen). La diferencia entre
     S2_sin_bloque_origen y S2_completo aisla la contribucion incremental
     de conocer el bloque de origen generalizado por si solo.

Una sola ejecucion (semilla=42), igual que el resto de analisis
diagnosticos de esta auditoria (sensibilidad de umbral, importancia por
permutacion) -- no reemplaza la comparacion primaria de 5 semillas de la
etapa 08, la complementa.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "03_ataque_reidentificacion" / "01_scripts"))

from anonymization_module import (  # noqa: E402
    CICIDS_RAW_ID_FEATURES,
    UNSW_RAW_ID_FEATURES,
    generalize_ip,
    generalize_port,
)
from attack_module import build_cicids_target, run_attack  # noqa: E402

STAGE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = STAGE_DIR / "02_outputs"
LOG_DIR = STAGE_DIR / "04_logs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

PREPROC_OUT = STAGE_DIR.parent / "02_preprocesamiento_baseline" / "02_outputs"
SEED = 42


def anonymize_columns(df, dataset):
    df = df.copy()
    if dataset == "CICIDS2017":
        ip_cols, port_cols = ["Source IP", "Destination IP"], ["Source Port", "Destination Port"]
        src_ip_col = "Source IP"
    else:
        ip_cols, port_cols = ["srcip", "dstip"], ["sport", "dsport"]
        src_ip_col = "srcip"
    anon_cols = []
    for c in ip_cols:
        df[f"{c}_anon"] = generalize_ip(df[c])
        anon_cols.append(f"{c}_anon")
    for c in port_cols:
        df[f"{c}_anon"] = generalize_port(df[c])
        anon_cols.append(f"{c}_anon")
    src_anon_col = f"{src_ip_col}_anon"
    return df, anon_cols, src_anon_col


def main():
    log_path = LOG_DIR / "02_matched_evidence_ablation_log.txt"
    with open(log_path, "w", encoding="utf-8") as fh:
        def log(msg):
            print(msg)
            fh.write(msg + "\n")

        results = {}
        for dataset_key, taxonomy_file, raw_id_cols, target_col in [
            ("CICIDS2017", "cicids2017_taxonomy.json", CICIDS_RAW_ID_FEATURES, "Source IP"),
            ("UNSW-NB15", "unsw_nb15_taxonomy.json", UNSW_RAW_ID_FEATURES, "srcip"),
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

            # S1: raw identifiers, exact target excluded (matches 04/08 scripts exactly)
            naive_cols = behavioral_cols + raw_id_cols
            evidence_s1 = [c for c in naive_cols if c != target_col]
            att_s1 = run_attack(train_df, test_df, evidence_s1, target_train, target_test, random_state=SEED)

            # S2: generalized identifiers
            train_anon, anon_cols, src_anon_col = anonymize_columns(train_df, dataset_key)
            test_anon, _, _ = anonymize_columns(test_df, dataset_key)

            evidence_s2_no_srcblock = behavioral_cols + [c for c in anon_cols if c != src_anon_col]
            att_s2_no_srcblock = run_attack(train_anon, test_anon, evidence_s2_no_srcblock,
                                             target_train, target_test, random_state=SEED)

            evidence_s2_full = behavioral_cols + anon_cols
            att_s2_full = run_attack(train_anon, test_anon, evidence_s2_full,
                                      target_train, target_test, random_state=SEED)

            log(f"[{dataset_key}] S1 (raw ids, exact src excluded): top1={att_s1.top1_accuracy:.4f}")
            log(f"[{dataset_key}] S2_no_srcblock (generalized dst+ports, NO src block; "
                f"matches S1 evidence composition): top1={att_s2_no_srcblock.top1_accuracy:.4f}")
            log(f"[{dataset_key}] S2_full (generalized dst+ports+src block): "
                f"top1={att_s2_full.top1_accuracy:.4f}")
            log(f"[{dataset_key}] isolated generalization effect (S1 -> S2_no_srcblock): "
                f"{att_s1.top1_accuracy - att_s2_no_srcblock.top1_accuracy:+.4f}")
            log(f"[{dataset_key}] incremental src-block contribution (S2_no_srcblock -> S2_full): "
                f"{att_s2_full.top1_accuracy - att_s2_no_srcblock.top1_accuracy:+.4f}")

            results[dataset_key] = {
                "s1_top1": att_s1.top1_accuracy,
                "s2_no_srcblock_top1": att_s2_no_srcblock.top1_accuracy,
                "s2_full_top1": att_s2_full.top1_accuracy,
                "isolated_generalization_effect_pp": att_s1.top1_accuracy - att_s2_no_srcblock.top1_accuracy,
                "incremental_srcblock_contribution_pp": att_s2_full.top1_accuracy - att_s2_no_srcblock.top1_accuracy,
            }

        with open(OUT_DIR / "matched_evidence_ablation.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        log("\nDone.")


if __name__ == "__main__":
    main()
