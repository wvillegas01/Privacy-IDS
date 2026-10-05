"""Modulo de anonimizacion de identificadores de red (Escenario 2), compartido
por las etapas 04 (necesita la version RAW de identificadores para fijar la
referencia de riesgo real del Escenario 1) y 05 (aplica la version
anonimizada).

Mecanismo: generalizacion tipo k-anonimato, no pseudonimizacion por hash.
- IP -> prefijo /16 (primeros dos octetos). Reduce cardinalidad real (varias
  IPs reales caen en el mismo grupo) en vez de solo re-etiquetar 1-a-1, que
  no tendria ningun costo de utilidad medible.
- Puerto -> banda IANA (well_known <1024, registered 1024-49151,
  dynamic >=49152). Mismo principio.

Distinto, por diseno, del Escenario 3 (etapa 06): ese elimina FEATURES DE
COMPORTAMIENTO segun el ranking del ataque de re-identificacion; este
generaliza columnas IDENTIFICADOR, sin tocar el comportamiento.
"""
import pandas as pd


def generalize_ip(series: pd.Series, octets: int = 2) -> pd.Series:
    """octets=2 -> /16 (default). octets=1 -> /8, the coarsest generalization
    that still keeps any structure (a single octet already spans 256 blocks).
    """
    def _bucket(ip):
        if not isinstance(ip, str):
            return "unknown"
        parts = ip.split(".")
        if len(parts) >= octets:
            return ".".join(parts[:octets])
        return "unknown"
    return series.astype(str).map(_bucket)


def generalize_port(series: pd.Series) -> pd.Series:
    def _bucket(p):
        try:
            p = int(p)
        except (ValueError, TypeError):
            return "unknown"
        if p < 0:
            return "unknown"
        if p < 1024:
            return "well_known"
        if p < 49152:
            return "registered"
        return "dynamic"
    return series.map(_bucket)


CICIDS_RAW_ID_FEATURES = ["Source IP", "Source Port", "Destination IP", "Destination Port"]
UNSW_RAW_ID_FEATURES = ["srcip", "sport", "dstip", "dsport"]


def add_raw_identifier_features(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Scenario 1 (naive baseline): raw identifiers used as-is, as extra
    categorical/numeric detection features. Excludes Flow ID / Timestamp /
    Stime / Ltime -- pure bookkeeping (a redundant concatenation string or
    an absolute wall-clock value), never used as a raw ML feature in IDS
    literature and with no legitimate detection value.
    """
    cols = CICIDS_RAW_ID_FEATURES if dataset == "CICIDS2017" else UNSW_RAW_ID_FEATURES
    return df[cols].copy(), cols


def add_anonymized_identifier_features(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Scenario 2: same identifier fields, but generalized (see module
    docstring) instead of raw or dropped.
    """
    if dataset == "CICIDS2017":
        out = pd.DataFrame(index=df.index)
        out["Source IP"] = generalize_ip(df["Source IP"])
        out["Destination IP"] = generalize_ip(df["Destination IP"])
        out["Source Port"] = generalize_port(df["Source Port"])
        out["Destination Port"] = generalize_port(df["Destination Port"])
        return out, ["Source IP", "Destination IP", "Source Port", "Destination Port"]
    else:
        out = pd.DataFrame(index=df.index)
        out["srcip"] = generalize_ip(df["srcip"])
        out["dstip"] = generalize_ip(df["dstip"])
        out["sport"] = generalize_port(df["sport"])
        out["dsport"] = generalize_port(df["dsport"])
        return out, ["srcip", "dstip", "sport", "dsport"]
