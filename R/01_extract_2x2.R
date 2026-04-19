# 01_extract_2x2.R — Load and validate 2x2 extraction
# Project 61: Rural-Calibrated Febrile-Infant Decision Support Tool

source("utils.R")

# Load extracted 2x2 data
cat("Loading extracted 2x2 data...\n")
df <- load_extracted_2x2()

cat(sprintf("Loaded %d rows across %d rules and %d cohorts\n",
            nrow(df), length(unique(df$rule)), length(unique(df$cohort))))

# Summary per rule
cat("\n--- Per-rule summary ---\n")
for (rule in unique(df$rule)) {
  sub <- df[df$rule == rule, ]
  cat(sprintf("  %s: %d cohorts, total n=%d, pooled prev=%.1f%%\n",
              rule, nrow(sub), sum(sub$n),
              100 * sum(sub$TP + sub$FN) / sum(sub$n)))
}

# Check for zero cells (will need continuity correction)
zero_cells <- df[df$TP == 0 | df$FP == 0 | df$FN == 0 | df$TN == 0, ]
if (nrow(zero_cells) > 0) {
  cat(sprintf("\nWARNING: %d rows have zero cells (continuity correction will be applied)\n",
              nrow(zero_cells)))
  print(zero_cells[, c("rule", "cohort", "TP", "FP", "FN", "TN")])
}

cat("\nValidation complete.\n")
