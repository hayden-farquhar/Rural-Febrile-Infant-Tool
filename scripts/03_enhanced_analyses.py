"""Missing-input degradation, CRP information value, age strata, decision curve, and rule comparison.

Inputs:  PECARN Biosignatures CSV files (see data/raw/README.md)
Outputs: results/enhanced_analyses_output.txt
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
from src.prediction_model import load_and_prepare_pecarn

os.makedirs("results", exist_ok=True)
out = open("results/enhanced_analyses_output.txt", "w")
def p(s=""): out.write(s + "\n"); out.flush()

base_path = "data/raw/pecarn_tig/Biosignatures_Full/CSV datasets"

p("Loading data...")
df = load_and_prepare_pecarn()
# ua_pos already built by load_and_prepare_pecarn
p(f"n={len(df)}, IBI={df.has_ibi.sum()} ({100*df.has_ibi.mean():.1f}%)")

# Load CRP
ob = pd.read_csv(f"{base_path}/labresults_otherblood.csv")
crp = ob[ob.BloodTest == 1][["PId", "BloodResult"]].rename(columns={"BloodResult": "crp"})
crp = crp.groupby("PId")["crp"].first().reset_index()
df = df.merge(crp, on="PId", how="left")

y = df["has_ibi"].astype(int)

# ═══════════════════════════════════════════════════════════════
p("\n" + "="*70)
p("ANALYSIS 1: MISSING-INPUT DEGRADATION CURVE")
p("="*70)

all_features = ["age_days", "temp_c", "wbc", "anc", "ua_pos"]
feature_labels = {"age_days": "Age", "temp_c": "Temperature", "wbc": "WBC",
                  "anc": "ANC", "ua_pos": "UA", "pct": "PCT", "crp": "CRP"}

# Full model with PCT (on complete subset)
pct_feats = all_features + ["pct"]
pct_complete = df.dropna(subset=pct_feats)
X_pct = pct_complete[pct_feats]; y_pct = pct_complete["has_ibi"].astype(int)
Xtr, Xte, ytr, yte = train_test_split(X_pct, y_pct, test_size=0.30, random_state=42, stratify=y_pct)
lr_full = LogisticRegression(max_iter=1000, random_state=42)
lr_full.fit(Xtr, ytr)
yp_full = lr_full.predict_proba(Xte)[:, 1]
auc_full = roc_auc_score(yte, yp_full)
lr1_full = (yp_full < 0.01).sum() / len(yp_full)
p(f"\nFull model (all + PCT): AUC={auc_full:.3f}, P<1%={lr1_full:.1%}")

# Systematically remove each feature
p(f"\nFeature removal impact (on PCT-complete subset, n={len(pct_complete)}):")
p(f"{'Removed':<15} {'AUC':>6} {'ΔAUC':>7} {'%P<1%':>7} {'Δ%P<1%':>8}")
p("-" * 50)
for feat in pct_feats:
    reduced = [f for f in pct_feats if f != feat]
    Xtr_r, Xte_r = Xtr[reduced], Xte[reduced]
    lr_r = LogisticRegression(max_iter=1000, random_state=42)
    lr_r.fit(Xtr_r, ytr)
    yp_r = lr_r.predict_proba(Xte_r)[:, 1]
    auc_r = roc_auc_score(yte, yp_r)
    lr1_r = (yp_r < 0.01).sum() / len(yp_r)
    p(f"{feature_labels.get(feat, feat):<15} {auc_r:>6.3f} {auc_r-auc_full:>+7.3f} {lr1_r:>6.1%} {lr1_r-lr1_full:>+7.1%}")

# Base model WITHOUT PCT
base_complete = df.dropna(subset=all_features)
X_base = base_complete[all_features]; y_base = base_complete["has_ibi"].astype(int)
Xtr_b, Xte_b, ytr_b, yte_b = train_test_split(X_base, y_base, test_size=0.30, random_state=42, stratify=y_base)
lr_base = LogisticRegression(max_iter=1000, random_state=42)
lr_base.fit(Xtr_b, ytr_b)
yp_base = lr_base.predict_proba(Xte_b)[:, 1]
auc_base = roc_auc_score(yte_b, yp_base)
lr1_base = (yp_base < 0.01).sum() / len(yp_base)
p(f"\nBase model (no PCT): AUC={auc_base:.3f}, P<1%={lr1_base:.1%}, n={len(base_complete)}")

# ═══════════════════════════════════════════════════════════════
p("\n" + "="*70)
p("ANALYSIS 2: CRP INFORMATION VALUE")
p("="*70)

crp_feats = all_features + ["crp"]
crp_complete = df.dropna(subset=crp_feats)
X_crp = crp_complete[crp_feats]; y_crp = crp_complete["has_ibi"].astype(int)
p(f"Patients with CRP: n={len(crp_complete)}, IBI={y_crp.sum()}")

if len(crp_complete) >= 50 and y_crp.sum() >= 3:
    Xtr_c, Xte_c, ytr_c, yte_c = train_test_split(X_crp, y_crp, test_size=0.30, random_state=42, stratify=y_crp)
    # With CRP
    lr_crp = LogisticRegression(max_iter=1000, random_state=42)
    lr_crp.fit(Xtr_c, ytr_c)
    yp_crp = lr_crp.predict_proba(Xte_c)[:, 1]
    auc_crp = roc_auc_score(yte_c, yp_crp)
    # Without CRP (same patients)
    lr_nocrp = LogisticRegression(max_iter=1000, random_state=42)
    lr_nocrp.fit(Xtr_c[all_features], ytr_c)
    yp_nocrp = lr_nocrp.predict_proba(Xte_c[all_features])[:, 1]
    auc_nocrp = roc_auc_score(yte_c, yp_nocrp)
    p(f"  Base (no CRP): AUC={auc_nocrp:.3f}")
    p(f"  With CRP:      AUC={auc_crp:.3f} (ΔAUC={auc_crp-auc_nocrp:+.3f})")

    # Both CRP and PCT?
    both_feats = all_features + ["crp", "pct"]
    both_complete = df.dropna(subset=both_feats)
    p(f"  Patients with both CRP+PCT: n={len(both_complete)}, IBI={both_complete.has_ibi.sum()}")
else:
    p("  Insufficient CRP data for sub-analysis")

# ═══════════════════════════════════════════════════════════════
p("\n" + "="*70)
p("ANALYSIS 3: AGE-STRATIFIED MODELS")
p("="*70)

for age_label, lo, hi in [("0-28d", 0, 28), ("29-60d", 29, 60)]:
    sub = base_complete[(base_complete.age_days >= lo) & (base_complete.age_days <= hi)]
    y_sub = sub["has_ibi"].astype(int)
    p(f"\n{age_label}: n={len(sub)}, IBI={y_sub.sum()} ({100*y_sub.mean():.1f}%)")

    if y_sub.sum() < 5:
        p("  Insufficient IBI cases for stratified analysis")
        continue

    feats_no_age = [f for f in all_features if f != "age_days"]
    Xtr_a, Xte_a, ytr_a, yte_a = train_test_split(
        sub[feats_no_age], y_sub, test_size=0.30, random_state=42, stratify=y_sub
    )
    lr_age = LogisticRegression(max_iter=1000, random_state=42)
    lr_age.fit(Xtr_a, ytr_a)
    yp_age = lr_age.predict_proba(Xte_a)[:, 1]
    auc_age = roc_auc_score(yte_a, yp_age)
    lr1_age = (yp_age < 0.01).sum() / len(yp_age)
    p(f"  AUC={auc_age:.3f}, %P<1%={lr1_age:.1%}")

    for thresh in [0.01, 0.02, 0.03]:
        below = yp_age < thresh
        ibi_missed = yte_a.values[below].sum()
        if below.sum() > 0:
            p(f"  P<{thresh:.0%}: {below.sum()}/{len(yp_age)} ({100*below.mean():.1f}%), IBI missed={ibi_missed} ({100*ibi_missed/max(below.sum(),1):.1f}%)")

# ═══════════════════════════════════════════════════════════════
p("\n" + "="*70)
p("ANALYSIS 4: DECISION CURVE ANALYSIS")
p("="*70)

# Net benefit = TP/N - FP/N × (pt/(1-pt))
# where pt = threshold probability
p(f"\nUsing base model (no PCT), test set n={len(yte_b)}, IBI={yte_b.sum()}")
p(f"{'Threshold':>10} {'Model NB':>10} {'Admit-all NB':>12} {'Model better?':>14}")
p("-" * 50)

for pt in [0.005, 0.01, 0.015, 0.02, 0.03, 0.05]:
    # Model
    pred_pos = yp_base >= pt
    tp = ((pred_pos) & (yte_b.values == 1)).sum()
    fp = ((pred_pos) & (yte_b.values == 0)).sum()
    n = len(yte_b)
    nb_model = tp/n - fp/n * (pt/(1-pt))

    # Admit all (treat all as positive)
    tp_all = yte_b.sum()
    fp_all = (yte_b == 0).sum()
    nb_all = tp_all/n - fp_all/n * (pt/(1-pt))

    better = "YES" if nb_model > nb_all else "no"
    p(f"{pt:>9.1%} {nb_model:>10.4f} {nb_all:>12.4f} {better:>14}")

# ═══════════════════════════════════════════════════════════════
p("\n" + "="*70)
p("ANALYSIS 5: HEAD-TO-HEAD vs BINARY RULES")
p("="*70)

# Apply Aronson binary rule to same patients as base model
from src.rules.aronson import AronsonInputs, apply as aronson_apply

aronson_preds = []
for _, row in base_complete.iterrows():
    if pd.isna(row["age_days"]):
        aronson_preds.append(None)
        continue
    r = aronson_apply(AronsonInputs(
        age_days=int(row["age_days"]), temp_c=row["temp_c"],
        ua_le_positive=bool(row["ua_pos"] > 0.5) if pd.notna(row["ua_pos"]) else None,
        ua_nitrites_positive=False,  # conservative
        anc=row["anc"] if pd.notna(row["anc"]) else None,
    ))
    aronson_preds.append(r.prediction if r.applicable else None)

base_complete = base_complete.copy()
base_complete["aronson_pred"] = aronson_preds
has_both = base_complete[base_complete["aronson_pred"].notna()].copy()
y_both = has_both["has_ibi"].astype(int)

# Aronson binary performance
aronson_binary = has_both["aronson_pred"].astype(int)
tp_a = ((aronson_binary == 1) & (y_both == 1)).sum()
fn_a = ((aronson_binary == 0) & (y_both == 1)).sum()
fp_a = ((aronson_binary == 1) & (y_both == 0)).sum()
tn_a = ((aronson_binary == 0) & (y_both == 0)).sum()
sens_a = tp_a / (tp_a + fn_a) if (tp_a + fn_a) > 0 else 0
spec_a = tn_a / (tn_a + fp_a) if (tn_a + fp_a) > 0 else 0
low_risk_a = (aronson_binary == 0).sum()
ibi_in_lr_a = fn_a

p(f"\nAronson binary (n={len(has_both)}, IBI={y_both.sum()}):")
p(f"  Sens={sens_a:.3f}, Spec={spec_a:.3f}")
p(f"  Low-risk: {low_risk_a} ({100*low_risk_a/len(has_both):.1f}%), IBI missed={ibi_in_lr_a}")

# Continuous model on same patients (at 1% threshold)
imp_both = SimpleImputer(strategy="median")
X_both = pd.DataFrame(imp_both.fit_transform(has_both[all_features]), columns=all_features)
# Use the previously trained base model
yp_both = lr_base.predict_proba(X_both)[:, 1]
low_risk_m = (yp_both < 0.01).sum()
ibi_in_lr_m = y_both.values[yp_both < 0.01].sum()

p(f"\nContinuous model at P<1% (same patients):")
p(f"  Low-risk: {low_risk_m} ({100*low_risk_m/len(has_both):.1f}%), IBI missed={ibi_in_lr_m}")

low_risk_m2 = (yp_both < 0.02).sum()
ibi_in_lr_m2 = y_both.values[yp_both < 0.02].sum()
p(f"\nContinuous model at P<2%:")
p(f"  Low-risk: {low_risk_m2} ({100*low_risk_m2/len(has_both):.1f}%), IBI missed={ibi_in_lr_m2}")

# ═══════════════════════════════════════════════════════════════
p("\n" + "="*70)
p("ANALYSIS 6: CALIBRATION")
p("="*70)

# Calibration curve for base model
prob_true, prob_pred = calibration_curve(yte_b, yp_base, n_bins=8, strategy="quantile")
p(f"\nCalibration curve (base model, test set):")
p(f"{'Predicted':>10} {'Observed':>10} {'n':>6}")
for pt, pp in zip(prob_pred, prob_true):
    p(f"{pp:>10.4f} {pt:>10.4f}")

# Brier score
brier = brier_score_loss(yte_b, yp_base)
p(f"\nBrier score: {brier:.4f}")

# Calibration slope and intercept
from sklearn.linear_model import LogisticRegression as LR2
logit_pred = np.log(np.clip(yp_base, 1e-6, 1-1e-6) / (1 - np.clip(yp_base, 1e-6, 1-1e-6)))
cal_model = LR2(max_iter=1000)
cal_model.fit(logit_pred.reshape(-1, 1), yte_b)
p(f"Calibration slope: {cal_model.coef_[0][0]:.3f} (ideal=1)")
p(f"Calibration intercept: {cal_model.intercept_[0]:.3f} (ideal=0)")

p("\n" + "="*70)
p("DONE")
p("="*70)
out.close()
print("Results written to results/enhanced_analyses_output.txt")
