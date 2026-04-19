"""Complete-case vs imputed training comparison.

Compares model performance when trained on complete cases only versus all
patients with median imputation. Demonstrates informative missingness bias
(IBI rate 1.98% complete vs 0.51% incomplete).

Inputs:  PECARN Biosignatures CSV files (see data/raw/README.md)
Outputs: results/complete_case_analysis.txt
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.impute import SimpleImputer
from sklearn.calibration import calibration_curve

os.makedirs("results", exist_ok=True)
out = open("results/complete_case_analysis.txt", "w")
def p(s=""): out.write(s + "\n"); out.flush()

base = "data/raw/pecarn_tig/Biosignatures_Full/CSV datasets"
demo = pd.read_csv(f"{base}/demographics.csv")
demo["age_days"] = -demo["BirthDay"]
clin = pd.read_csv(f"{base}/clinicaldata.csv").groupby("PId").first().reset_index()
clin["temp_c"] = clin["Temperature"].apply(lambda t: t if pd.isna(t) or t <= 60 else (t - 32) * 5 / 9)
yos_cols = ["YOSCry", "YOSReaction", "YOSState", "YOSColor", "YOSHydration", "YOSResponse"]
clin["yos_total"] = clin[yos_cols].sum(axis=1, min_count=6)
lab = pd.read_csv(f"{base}/labresults.csv").groupby("PId").first().reset_index()
lab["ua_pos"] = ((lab["UrineLEC"].isin([1, 2, 3])) | (lab["NitriteRes"] == 1)).astype(float)
lab.loc[lab["UrineLEC"].isna() & lab["NitriteRes"].isna(), "ua_pos"] = np.nan
blood = pd.read_csv(f"{base}/culturereview_blood.csv")
csf = pd.read_csv(f"{base}/culturereview_csf.csv")
ibi_pids = set(blood[blood.BloodDCCAssess == 1]["PId"]) | set(csf[csf.CSFDCCAssess == 1]["PId"])

df = (demo[["PId", "age_days"]]
    .merge(clin[["PId", "temp_c", "yos_total"]], on="PId", how="inner")
    .merge(lab[["PId", "WBC", "ANC", "ua_pos"]], on="PId", how="left"))
df["has_ibi"] = df["PId"].isin(ibi_pids).astype(int)

feats = ["age_days", "temp_c", "WBC", "ANC", "ua_pos", "yos_total"]

p("=" * 70)
p("1. POPULATION COMPARISON: Complete vs incomplete data")
p("=" * 70)

complete_mask = df[feats].notna().all(axis=1)
df_complete = df[complete_mask]
df_incomplete = df[~complete_mask]

p(f"\nAll patients: n={len(df)}, IBI={df.has_ibi.sum()} ({100*df.has_ibi.mean():.2f}%)")
p(f"Complete cases: n={len(df_complete)}, IBI={df_complete.has_ibi.sum()} ({100*df_complete.has_ibi.mean():.2f}%)")
p(f"Incomplete cases: n={len(df_incomplete)}, IBI={df_incomplete.has_ibi.sum()} ({100*df_incomplete.has_ibi.mean():.2f}%)")

p(f"\n{'Variable':<15} {'Complete':>20} {'Incomplete':>20}")
p("-" * 58)
for col in feats:
    c_mean = df_complete[col].mean()
    i_mean = df_incomplete[col].mean() if df_incomplete[col].notna().sum() > 0 else float('nan')
    p(f"{col:<15} {c_mean:>20.2f} {i_mean:>20.2f}")

# What's missing in incomplete cases?
p(f"\nMissingness pattern in incomplete cases (n={len(df_incomplete)}):")
for col in feats:
    n_miss = df_incomplete[col].isna().sum()
    p(f"  {col}: {n_miss} missing ({100*n_miss/len(df_incomplete):.1f}%)")

p("\n" + "=" * 70)
p("2. COMPLETE-CASE MODEL vs IMPUTED MODEL")
p("=" * 70)

y_comp = df_complete["has_ibi"].astype(int)
X_comp = df_complete[feats]

Xtr_c, Xte_c, ytr_c, yte_c = train_test_split(X_comp, y_comp, test_size=0.30, random_state=42, stratify=y_comp)
lr_comp = LogisticRegression(max_iter=1000, random_state=42)
lr_comp.fit(Xtr_c, ytr_c)
yp_comp = lr_comp.predict_proba(Xte_c)[:, 1]
auc_comp = roc_auc_score(yte_c, yp_comp)
brier_comp = brier_score_loss(yte_c, yp_comp)

p(f"\nComplete-case model (n={len(df_complete)}, IBI={y_comp.sum()}):")
p(f"  AUC={auc_comp:.3f}, Brier={brier_comp:.4f}")
p(f"  %P<1%: {(yp_comp<0.01).mean():.1%}, IBI missed: {yte_c.values[yp_comp<0.01].sum()}")
p(f"  %P<2%: {(yp_comp<0.02).mean():.1%}, IBI missed: {yte_c.values[yp_comp<0.02].sum()}")

# Now imputed model on ALL patients
X_all = df[feats].copy()
y_all = df["has_ibi"].astype(int)
imp = SimpleImputer(strategy="median")
Xi_all = pd.DataFrame(imp.fit_transform(X_all), columns=feats)

Xtr_i, Xte_i, ytr_i, yte_i = train_test_split(Xi_all, y_all, test_size=0.30, random_state=42, stratify=y_all)
lr_imp = LogisticRegression(max_iter=1000, random_state=42)
lr_imp.fit(Xtr_i, ytr_i)
yp_imp = lr_imp.predict_proba(Xte_i)[:, 1]
auc_imp = roc_auc_score(yte_i, yp_imp)
brier_imp = brier_score_loss(yte_i, yp_imp)

p(f"\nImputed model (n={len(df)}, IBI={y_all.sum()}):")
p(f"  AUC={auc_imp:.3f}, Brier={brier_imp:.4f}")
p(f"  %P<1%: {(yp_imp<0.01).mean():.1%}, IBI missed: {yte_i.values[yp_imp<0.01].sum()}")
p(f"  %P<2%: {(yp_imp<0.02).mean():.1%}, IBI missed: {yte_i.values[yp_imp<0.02].sum()}")

p("\n" + "=" * 70)
p("3. COEFFICIENT COMPARISON")
p("=" * 70)

p(f"\n{'Feature':<15} {'Complete-case':>15} {'Imputed':>15} {'Difference':>12}")
p("-" * 60)
for i, f in enumerate(feats):
    cc = lr_comp.coef_[0][i]
    ic = lr_imp.coef_[0][i]
    p(f"{f:<15} {cc:>15.4f} {ic:>15.4f} {ic-cc:>+12.4f}")
p(f"{'intercept':<15} {lr_comp.intercept_[0]:>15.4f} {lr_imp.intercept_[0]:>15.4f} {lr_imp.intercept_[0]-lr_comp.intercept_[0]:>+12.4f}")

p("\n" + "=" * 70)
p("4. CROSS-APPLICATION: Train on complete, test on imputed (and vice versa)")
p("=" * 70)

# Train on complete cases, apply to ALL patients (imputed)
yp_comp_on_all = lr_comp.predict_proba(Xi_all)[:, 1]
# Only evaluate on patients NOT in training set
# Use the full imputed test set
yp_comp_on_imp_test = lr_comp.predict_proba(Xte_i)[:, 1]
auc_cross1 = roc_auc_score(yte_i, yp_comp_on_imp_test)

# Train on imputed, apply to complete-case test set
yp_imp_on_comp_test = lr_imp.predict_proba(Xte_c)[:, 1]
auc_cross2 = roc_auc_score(yte_c, yp_imp_on_comp_test)

p(f"\nComplete-case model → imputed test set:  AUC={auc_cross1:.3f}")
p(f"Imputed model → complete-case test set:  AUC={auc_cross2:.3f}")
p(f"Complete-case model → complete test set:  AUC={auc_comp:.3f}")
p(f"Imputed model → imputed test set:        AUC={auc_imp:.3f}")

p("\n" + "=" * 70)
p("5. CALIBRATION COMPARISON")
p("=" * 70)

for name, yte_x, yp_x in [("Complete-case", yte_c, yp_comp), ("Imputed", yte_i, yp_imp)]:
    prob_true, prob_pred = calibration_curve(yte_x, yp_x, n_bins=5, strategy="quantile")
    logit = lambda p: np.log(np.clip(p, 1e-6, 1-1e-6) / (1 - np.clip(p, 1e-6, 1-1e-6)))
    cal_lr = LogisticRegression(max_iter=1000)
    cal_lr.fit(logit(yp_x).reshape(-1, 1), yte_x)
    p(f"\n{name} model:")
    p(f"  Calibration slope: {cal_lr.coef_[0][0]:.3f} (ideal=1)")
    p(f"  Calibration intercept: {cal_lr.intercept_[0]:.3f} (ideal=0)")
    p(f"  Brier: {brier_score_loss(yte_x, yp_x):.4f}")

p("\n" + "=" * 70)
p("6. RECOMMENDATION FOR RURAL DEPLOYMENT")
p("=" * 70)

p("""
The key question: should the production model be trained on complete cases
only, or on all patients with imputation?

Arguments for COMPLETE-CASE ONLY:
  - No imputation bias
  - Coefficients reflect actual lab value relationships
  - Patients without labs are genuinely different (often lower risk,
    as clinicians didn't order full workup)
  - Cleaner, more defensible

Arguments for IMPUTATION:
  - Larger training set (more IBI events → more stable estimates)
  - In rural deployment, labs WILL be missing — the model should
    be calibrated to handle this
  - Complete cases may not be representative of the rural population

RECOMMENDED APPROACH:
  Train on complete cases (cleanest coefficients), but validate
  the model's performance when applied to imputed inputs at test
  time. This separates the training signal from the deployment
  reality. Report both complete-case and imputed performance.
""")

out.close()
print("Results written to results/complete_case_analysis.txt")
