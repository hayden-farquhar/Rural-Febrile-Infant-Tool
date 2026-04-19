"""Shared utilities for febrile-infant decision support tool."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

# Clinical thresholds (pre-registered)
IBI_TRANSFER_THRESHOLD = 0.01  # 1% IBI probability

# Conformal coverage targets (pre-registered)
COVERAGE_TARGETS = [0.90, 0.95]
ALPHA_LEVELS = [0.05, 0.10, 0.20]

# Age tertiles for Mondrian strata (pre-registered)
AGE_TERTILE_BOUNDS = [(0, 28), (29, 60), (61, 89)]
AGE_TERTILE_LABELS = ["0-28d", "29-60d", "61-89d"]

# Number of imputations (pre-registered)
N_IMPUTATIONS = 20

# Minimum calibration cases per Mondrian stratum before merging
MIN_STRATUM_SIZE = 20
