# utils.R — Shared utilities for R meta-analysis scripts
#
# Run R scripts from the R/ directory:
#   cd R/
#   Rscript 01_extract_2x2.R

library(mada)
library(meta)
library(metafor)

# Paths assume working directory is R/ (one level below repository root)
DATA_DIR <- file.path(dirname(getwd()), "data")
RAW_DIR <- file.path(DATA_DIR, "raw")
INTERIM_DIR <- file.path(DATA_DIR, "interim")
RESULTS_DIR <- file.path(dirname(getwd()), "results")

# Ensure output directories exist
dir.create(RESULTS_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(RESULTS_DIR, "tables"), showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(RESULTS_DIR, "figures"), showWarnings = FALSE, recursive = TRUE)
dir.create(INTERIM_DIR, showWarnings = FALSE, recursive = TRUE)

load_extracted_2x2 <- function(path = file.path(RAW_DIR, "extracted_2x2.csv")) {
  df <- read.csv(path, stringsAsFactors = FALSE)
  required_cols <- c("rule", "cohort", "threshold", "TP", "FP", "FN", "TN")
  missing <- setdiff(required_cols, names(df))
  if (length(missing) > 0) {
    stop(paste("Missing required columns:", paste(missing, collapse = ", ")))
  }
  # Validate no negative counts
  count_cols <- c("TP", "FP", "FN", "TN")
  for (col in count_cols) {
    if (any(df[[col]] < 0, na.rm = TRUE)) {
      stop(paste("Negative values found in", col))
    }
  }
  df$n <- df$TP + df$FP + df$FN + df$TN
  df$sensitivity <- df$TP / (df$TP + df$FN)
  df$specificity <- df$TN / (df$TN + df$FP)
  df
}
