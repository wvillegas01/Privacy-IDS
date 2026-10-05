# Code and derived results: host re-identification risk in intrusion detection datasets

This repository accompanies the manuscript *Measuring Host Re-identification Risk in Intrusion Detection
Datasets: An Attack-Based Comparison of Privacy Techniques* (Jaramillo-Alcazar, Mera Navarrete, Gutierrez,
Villegas-Ch). It contains the analysis scripts and the derived result files behind every table and figure.
The raw benchmark datasets are **not** redistributed; they are public and must be downloaded from their owners.

## Contents

| Path | Description |
|---|---|
| `pipeline/<stage>/01_scripts/` | Python scripts of each analysis stage |
| `pipeline/<stage>/02_outputs/` | Derived results (JSON) produced by those scripts |
| `figures/make_figures.py` | Regenerates Figures 1-4 as vector PDFs from the JSON results |
| `requirements.txt` | Library versions used (Python 3.8) |

Stage folders keep their original names: `01_inventario_datos` (data inventory), `02_preprocesamiento_baseline`
(cleaning, identifier/behavior taxonomy, split), `03_ataque_reidentificacion` (attack module and baseline attack),
`04_escenario1_baseline` (S1 and detection module), `05_escenario2_anonimizacion` (S2 and matched-evidence
ablation), `06_escenario3_reduccion_features` (S3 and threshold sensitivity), `07_escenario4_dp_ruido` (S4),
`08_comparacion_transversal` (five-condition comparison over 5 seeds), `10_amenazas_validez_limitaciones`
(host-based partition robustness check).

## Data (not included)

Download the following public variants, which retain raw network identifiers:

* CICIDS2017, `GeneratedLabelledFlows.zip` (the `TrafficLabelling` CSV files), Canadian Institute for
  Cybersecurity, University of New Brunswick: https://www.unb.ca/cic/datasets/ids-2017.html
* UNSW-NB15, the four raw files `UNSW-NB15_1.csv` to `UNSW-NB15_4.csv` plus `NUSW-NB15_features.csv`, UNSW
  Canberra Cyber Range Lab: https://research.unsw.edu.au/projects/unsw-nb15-dataset

Please follow each dataset owner's terms of use and citation requirements.

## Configuration

Set these environment variables (defaults are relative paths under `data/`):

* `CICIDS2017_RAW_DIR`: folder with the CICIDS2017 `TrafficLabelling` CSV files
* `UNSW_NB15_RAW_DIR`: folder with the UNSW-NB15 raw CSV files and `NUSW-NB15_features.csv`
* `PRIVACY_IDS_HEAVY_DIR`: folder for large intermediate files (cleaned parquet files and splits)

## Running the pipeline

Scripts import their sibling modules by relative path, so keep the folder structure unchanged. Run in order
from the repository root (all seeds are fixed; the base seed is 42):

1. `pipeline/01_inventario_datos/01_scripts/02_inventario_unsw_raw.py`, `03_inventario_cicids_raw.py`
2. `pipeline/02_preprocesamiento_baseline/01_scripts/01_clean_cicids2017.py`, `02_clean_unsw_nb15.py`
3. `pipeline/03_ataque_reidentificacion/01_scripts/01_run_baseline_attack.py`, `02_permutation_importance_check.py`
4. `pipeline/04_escenario1_baseline/01_scripts/01_run_baseline_scenario.py`
5. `pipeline/05_escenario2_anonimizacion/01_scripts/01_run_anonymization_scenario.py`, `02_matched_evidence_ablation.py`
6. `pipeline/06_escenario3_reduccion_features/01_scripts/01_run_feature_reduction_scenario.py`, `02_threshold_sensitivity.py`
7. `pipeline/07_escenario4_dp_ruido/01_scripts/01_run_dp_scenario.py`
8. `pipeline/08_comparacion_transversal/01_scripts/01_run_cross_scenario_comparison.py`, `02_add_s0_scenario.py`, `03_update_s4_clipped_dp.py`
9. `pipeline/10_amenazas_validez_limitaciones/01_scripts/01_host_split_robustness.py`
10. `python figures/make_figures.py` (set `FIG_OUT` to choose the output folder; no raw data required)

Some scripts write log files into a `04_logs` folder next to `01_scripts`; create it if it is missing.

## Where each manuscript element comes from

| Manuscript element | Result file |
|---|---|
| Table 1 (dataset characteristics) | stage 01 `*_raw_profile.json` and stage 02 `*_taxonomy.json` (cleaning logs are written to `04_logs`) |
| Table 3 (five-condition comparison), Figure 4 | `08_comparacion_transversal/02_outputs/cross_scenario_comparison.json` |
| Table 4 (feature-reduction thresholds) | `06_escenario3_reduccion_features/02_outputs/threshold_sensitivity.json` |
| Table 5 (host-based partition check) | `10_amenazas_validez_limitaciones/02_outputs/host_split_robustness.json` |
| Figure 2 (attack feature importance) | `03_ataque_reidentificacion/02_outputs/baseline_attack_results.json` |
| Figure 3 (budget sweep) | `07_escenario4_dp_ruido/02_outputs/scenario4_dp_results.json` |
| S1 vs S2 decomposition | `05_escenario2_anonimizacion/02_outputs/matched_evidence_ablation.json` |

## Notes

* S4 is a clipped Laplace perturbation with a nominal composed budget; it is **not** a formal differential
  privacy guarantee for the joint release (see the manuscript).
* Scripts, module docstrings and comments are partly in Spanish.
* The analysis code was developed with the assistance of a generative AI tool (Claude, Anthropic), and was
  reviewed by the authors.

## License and citation

Code and derived results are released under the MIT License (see `LICENSE`). Citation: to be added with the
DOI of the published article.
