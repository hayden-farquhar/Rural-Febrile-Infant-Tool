# 02_hsroc_bivariate.R — Bivariate HSROC meta-analysis per rule
# Project 61: Rural-Calibrated Febrile-Infant Decision Support Tool

source("utils.R")

df <- load_extracted_2x2()

rules_with_multiple_cohorts <- names(which(table(df$rule) >= 2))
cat(sprintf("Rules with >=2 cohorts for bivariate model: %s\n",
            paste(rules_with_multiple_cohorts, collapse = ", ")))

fits <- list()
results_table <- data.frame()

for (rule in rules_with_multiple_cohorts) {
  sub <- df[df$rule == rule, ]
  cat(sprintf("\n=== %s (%d cohorts) ===\n", rule, nrow(sub)))

  # Fit bivariate model (Reitsma)
  fit <- reitsma(sub[, c("TP", "FP", "FN", "TN")])
  fits[[rule]] <- fit

  s <- summary(fit)
  coefs <- s$coefficients

  # Extract pooled sensitivity and specificity (back-transformed)
  pooled_sens <- coefs["sensitivity", "Estimate"]
  sens_lb <- coefs["sensitivity", "95%ci.lb"]
  sens_ub <- coefs["sensitivity", "95%ci.ub"]

  pooled_fpr <- coefs["false pos. rate", "Estimate"]
  fpr_lb <- coefs["false pos. rate", "95%ci.lb"]
  fpr_ub <- coefs["false pos. rate", "95%ci.ub"]
  pooled_spec <- 1 - pooled_fpr
  spec_lb <- 1 - fpr_ub  # note: CI bounds flip
  spec_ub <- 1 - fpr_lb

  cat(sprintf("  Pooled sensitivity: %.3f (%.3f-%.3f)\n", pooled_sens, sens_lb, sens_ub))
  cat(sprintf("  Pooled specificity: %.3f (%.3f-%.3f)\n", pooled_spec, spec_lb, spec_ub))
  cat(sprintf("  AUC: %.3f\n", s$AUC))

  # Store results
  results_table <- rbind(results_table, data.frame(
    rule = rule,
    n_cohorts = nrow(sub),
    total_n = sum(sub$n),
    pooled_sens = round(pooled_sens, 4),
    sens_95ci = sprintf("%.3f-%.3f", sens_lb, sens_ub),
    pooled_spec = round(pooled_spec, 4),
    spec_95ci = sprintf("%.3f-%.3f", spec_lb, spec_ub),
    AUC = round(as.numeric(s$AUC)[2], 4),
    stringsAsFactors = FALSE
  ))

  # HSROC curve
  png(file.path(RESULTS_DIR, "figures", sprintf("hsroc_%s.png", gsub(" ", "_", rule))),
      width = 800, height = 800, res = 150)
  plot(fit, main = sprintf("HSROC: %s", rule),
       sroclwd = 2, predict = TRUE)
  points(fpr(sub[, c("TP", "FP", "FN", "TN")]),
         sens(sub[, c("TP", "FP", "FN", "TN")]),
         pch = 19, cex = 1.5)
  dev.off()
  cat(sprintf("  HSROC plot saved to results/figures/hsroc_%s.png\n", gsub(" ", "_", rule)))
}

# Report single-cohort rules
single_rules <- names(which(table(df$rule) == 1))
if (length(single_rules) > 0) {
  cat("\n=== Single-cohort rules (no pooling) ===\n")
  for (rule in single_rules) {
    sub <- df[df$rule == rule, ]
    cat(sprintf("  %s: sens=%.3f, spec=%.3f (single study, n=%d, %s)\n",
                rule, sub$sensitivity, sub$specificity, sub$n, sub$cohort))

    results_table <- rbind(results_table, data.frame(
      rule = rule,
      n_cohorts = 1,
      total_n = sub$n,
      pooled_sens = round(sub$sensitivity, 4),
      sens_95ci = "single study",
      pooled_spec = round(sub$specificity, 4),
      spec_95ci = "single study",
      AUC = NA,
      stringsAsFactors = FALSE
    ))
  }
}

# Print summary table
cat("\n=== Summary table ===\n")
print(results_table, row.names = FALSE)

# Save results
write.csv(results_table, file.path(RESULTS_DIR, "tables", "hsroc_pooled_results.csv"),
          row.names = FALSE)
cat("\nResults saved to results/tables/hsroc_pooled_results.csv\n")

# Save fits
saveRDS(fits, file.path(INTERIM_DIR, "hsroc_fits.rds"))
cat("Fit objects saved to data/interim/hsroc_fits.rds\n")
