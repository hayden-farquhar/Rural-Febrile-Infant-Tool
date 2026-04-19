# Rural-calibrated febrile infant decision support tool

Code repository for: **Rural-calibrated febrile infant decision support tool: bivariate meta-analysis of published decision rules with individual-level prediction modelling**

Hayden Farquhar MBBS MPHTM

ORCID: [0009-0002-6226-440X](https://orcid.org/0009-0002-6226-440X)

Pre-registration: [OSF dq5n8](https://osf.io/dq5n8/)

Preprint: to be posted

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19649953.svg)](https://doi.org/10.5281/zenodo.19649953)

## Overview

This repository contains the analysis code for a study combining (1) bivariate HSROC meta-analysis of seven published febrile infant decision rules with (2) individual-level prediction modelling using the PECARN Biosignatures public-use dataset. The model generates continuous IBI (invasive bacterial infection) probability estimates for febrile infants aged 0-60 days using clinical and laboratory data routinely available in rural emergency departments without procalcitonin. A Streamlit decision support tool is included.

## Data Sources

| Source | URL | Access | License |
|--------|-----|--------|---------|
| PECARN Biosignatures public-use dataset | https://pecarn.org/datasets/ | Registration + Data Use Agreement | PECARN DUA |
| Extracted 2x2 tables (this study) | Included in `data/raw/extracted_2x2.csv` | Open | CC-BY 4.0 |

The PECARN Biosignatures dataset cannot be redistributed. To reproduce the prediction model analyses, download the dataset from the PECARN Data Coordinating Center under a Research Data Use Agreement. Place the extracted CSV files at:

```
data/raw/pecarn_tig/Biosignatures_Full/CSV datasets/
  ├── demographics.csv
  ├── clinicaldata.csv
  ├── labresults.csv
  ├── pctdata.csv
  ├── labresults_otherblood.csv
  ├── culturereview_blood.csv
  └── culturereview_csf.csv
```

## Requirements

### R (meta-analysis)

R >= 4.4 with packages: `mada`, `meta`, `metafor`

```r
install.packages(c("mada", "meta", "metafor"))
```

### Python (prediction model and tool)

Python >= 3.11

```bash
pip install -r requirements.txt
```

## Reproduction

The analyses have two independent pipelines: R (meta-analysis) and Python (prediction model). They can be run in either order.

### Meta-analysis (R)

Run from the `R/` directory:

```bash
cd R/
Rscript 01_extract_2x2.R
Rscript 02_hsroc_bivariate.R
Rscript 03_sensitivity_meta.R
```

Outputs: `results/tables/hsroc_pooled_results.csv`, `results/figures/hsroc_*.png`

### Prediction model (Python)

Run from the repository root:

```bash
# Train and save model
python -m src.prediction_model

# Validation and analyses (each writes to results/)
python scripts/02_model_validation.py
python scripts/03_enhanced_analyses.py
python scripts/04_complete_case_analysis.py
python scripts/05_missingness_confound.py
python scripts/06_threshold_and_misses.py

# Generate manuscript figures
python scripts/01_generate_figures.py
```

### Interactive tool

```bash
streamlit run app/streamlit_app.py
```

### Tests

```bash
python -m pytest tests/ -v
```

**Estimated total runtime:** ~5 minutes on a standard laptop (no GPU required).

## Script Descriptions

| Script | Description | Inputs | Outputs |
|--------|-------------|--------|---------|
| `R/01_extract_2x2.R` | Load and validate 2x2 extraction data | `data/raw/extracted_2x2.csv` | Console validation summary |
| `R/02_hsroc_bivariate.R` | Bivariate HSROC meta-analysis per rule | 2x2 data | `results/tables/hsroc_pooled_results.csv`, HSROC curve PNGs |
| `R/03_sensitivity_meta.R` | Leave-one-out and subgroup sensitivity analyses | 2x2 data, HSROC fits | Console output |
| `src/prediction_model.py` | Production model v6 (train, predict, save/load) | PECARN CSV files | `data/interim/prediction_model_v6.joblib` |
| `scripts/01_generate_figures.py` | Generate manuscript figures 1-5 | PECARN data | `outputs/figures/figure*.png` |
| `scripts/02_model_validation.py` | Bootstrap validation, calibration, decision curve | PECARN data | `results/v6_validation_output.txt` |
| `scripts/03_enhanced_analyses.py` | Missing-input degradation, CRP analysis, age strata | PECARN data | `results/enhanced_analyses_output.txt` |
| `scripts/04_complete_case_analysis.py` | Complete-case vs imputed training comparison | PECARN data | `results/complete_case_analysis.txt` |
| `scripts/05_missingness_confound.py` | Missingness indicator confounding analysis | PECARN data | `results/missingness_confound_output.txt` |
| `scripts/06_threshold_and_misses.py` | Missed IBI case analysis and threshold evaluation | PECARN data | `results/threshold_and_misses_output.txt` |
| `app/streamlit_app.py` | Interactive decision support tool | Trained model | Streamlit web interface |

## Outputs

| File | Paper reference |
|------|----------------|
| `results/tables/hsroc_pooled_results.csv` | Table 1 |
| `results/figures/hsroc_Aronson.png` | eFigure 1 |
| `results/figures/hsroc_PECARN.png` | eFigure 1 |
| `outputs/figures/figure1_flow.png` | Figure 1 |
| `outputs/figures/figure2_calibration.png` | Figure 2 |
| `outputs/figures/figure3_decision_curve.png` | Figure 3 |
| `outputs/figures/figure4_risk_tiers.png` | Figure 4 |
| `outputs/figures/figure5_degradation.png` | Figure 5 |
| `results/v6_validation_output.txt` | Results: Discrimination, Calibration, Four-tier classification |
| `results/enhanced_analyses_output.txt` | Table 4, Results: Missing-input degradation |
| `results/complete_case_analysis.txt` | eTable 3 |
| `results/missingness_confound_output.txt` | Methods: Predictor selection (missingness discussion) |
| `results/threshold_and_misses_output.txt` | eTable 5 |

## Citation

If you use this code, please cite:

```
Farquhar H. Rural-calibrated febrile infant decision support tool: bivariate
meta-analysis of published decision rules with individual-level prediction
modelling. [Preprint]. 2026. Code: https://doi.org/10.5281/zenodo.19649953
```

## License

Code: MIT License
Data and documentation: CC-BY 4.0

See [LICENSE](LICENSE) for full text.
