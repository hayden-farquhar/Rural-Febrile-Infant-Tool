# 03_sensitivity_meta.R — Leave-one-out and subgroup sensitivity analyses
# Project 61: Rural-Calibrated Febrile-Infant Decision Support Tool

source("utils.R")

df <- load_extracted_2x2()
fits <- readRDS(file.path(INTERIM_DIR, "hsroc_fits.rds"))

rules_with_multiple_cohorts <- names(fits)

for (rule in rules_with_multiple_cohorts) {
  sub <- df[df$rule == rule, ]
  if (nrow(sub) < 3) {
    cat(sprintf("Skipping leave-one-out for %s (only %d cohorts)\n", rule, nrow(sub)))
    next
  }

  cat(sprintf("\n=== Leave-one-out: %s ===\n", rule))

  for (i in seq_len(nrow(sub))) {
    excluded <- sub$cohort[i]
    sub_loo <- sub[-i, ]
    fit_loo <- reitsma(sub_loo[, c("TP", "FP", "FN", "TN")])
    s <- summary(fit_loo)
    cat(sprintf("  Excluding %s: sens=%.3f, spec=%.3f\n",
                excluded,
                s$coefficients["tsens", "Estimate"],
                s$coefficients["tfpr", "Estimate"]))
  }
}

# Subgroup by era (pre-2015 vs 2015+) if publication_year available
if ("publication_year" %in% names(df)) {
  cat("\n=== Subgroup by publication era ===\n")
  for (rule in rules_with_multiple_cohorts) {
    sub <- df[df$rule == rule, ]
    pre <- sub[sub$publication_year < 2015, ]
    post <- sub[sub$publication_year >= 2015, ]

    if (nrow(pre) >= 2 && nrow(post) >= 2) {
      fit_pre <- reitsma(pre[, c("TP", "FP", "FN", "TN")])
      fit_post <- reitsma(post[, c("TP", "FP", "FN", "TN")])
      cat(sprintf("  %s pre-2015 (%d): sens=%.3f\n", rule, nrow(pre),
                  summary(fit_pre)$coefficients["tsens", "Estimate"]))
      cat(sprintf("  %s 2015+ (%d): sens=%.3f\n", rule, nrow(post),
                  summary(fit_post)$coefficients["tsens", "Estimate"]))
    } else {
      cat(sprintf("  %s: insufficient studies for era subgroup\n", rule))
    }
  }
} else {
  cat("\nNote: publication_year not in extracted data; era subgroup analysis skipped.\n")
}

# Subgroup by age stratum if age_group available
if ("age_group" %in% names(df)) {
  cat("\n=== Subgroup by age stratum ===\n")
  for (age in unique(df$age_group)) {
    sub_age <- df[df$age_group == age, ]
    cat(sprintf("  Age group: %s (%d rows)\n", age, nrow(sub_age)))
  }
} else {
  cat("\nNote: age_group not in extracted data; age subgroup analysis skipped.\n")
}

cat("\nSensitivity analyses complete.\n")
