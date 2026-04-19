"""Model validation: bootstrap, calibration, decision curve, and age-stratified performance.

Inputs:  PECARN Biosignatures CSV files (see data/raw/README.md)
Outputs: results/v6_validation_output.txt
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy.stats import chi2 as chi2_dist
import warnings
warnings.filterwarnings("ignore")

os.makedirs("results", exist_ok=True)
out = open("results/v6_validation_output.txt", "w")
def p(s=""): out.write(s + "\n"); out.flush()

from src.prediction_model import load_and_prepare_pecarn, FEATURES

df = load_and_prepare_pecarn()
base_dir = "data/raw/pecarn_tig/Biosignatures_Full/CSV datasets"
clin = pd.read_csv(f"{base_dir}/clinicaldata.csv").groupby("PId").first().reset_index()
yos_cols = ["YOSCry", "YOSReaction", "YOSState", "YOSColor", "YOSHydration", "YOSResponse"]
clin["yos_total"] = clin[yos_cols].sum(axis=1, min_count=6)
df = df.merge(clin[["PId", "yos_total"]], on="PId", how="left", suffixes=("", "_c"))
if "yos_total_c" in df.columns:
    df["yos_total"] = df["yos_total"].fillna(df["yos_total_c"])
    df.drop(columns=["yos_total_c"], inplace=True)

complete = df.dropna(subset=["age_days", "temp_c", "wbc", "anc", "ua_pos", "yos_total"]).copy()
complete["age_young"] = (complete["age_days"] <= 14).astype(float)
y = complete["has_ibi"].astype(int)

p(f"v6 POOLED MODEL VALIDATION")
p(f"Complete cases: n={len(complete)}, IBI={y.sum()} ({100*y.mean():.1f}%)")
p(f"Features: {FEATURES}")

Xtr, Xte, ytr, yte = train_test_split(complete[FEATURES], y, test_size=0.30, random_state=42, stratify=y)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(Xtr, ytr)
yp = lr.predict_proba(Xte)[:, 1]

# ── Discrimination ────────────────────────────────────────
auc = roc_auc_score(yte, yp)
disc = yp[yte.values == 1].mean() - yp[yte.values == 0].mean()
p(f"\nDISCRIMINATION:")
p(f"  AUC: {auc:.3f}")
p(f"  Discrimination slope: {disc:.4f}")

# ── Calibration ───────────────────────────────────────────
brier = brier_score_loss(yte, yp)
obs = yte.sum(); exp = yp.sum()
oe = obs / exp if exp > 0 else 0
logit_p = np.log(np.clip(yp, 1e-6, 1-1e-6) / (1 - np.clip(yp, 1e-6, 1-1e-6)))
cal_lr = LogisticRegression(max_iter=1000)
cal_lr.fit(logit_p.reshape(-1, 1), yte)
n_groups = 8
sorted_idx = np.argsort(yp)
groups = np.array_split(sorted_idx, n_groups)
hl_stat = 0
for g in groups:
    o = yte.values[g].sum(); e = yp[g].sum(); n = len(g)
    if e > 0 and (n - e) > 0: hl_stat += (o - e)**2 / (e * (1 - e/n))
hl_p = 1 - chi2_dist.cdf(hl_stat, n_groups - 2)

p(f"\nCALIBRATION:")
p(f"  Brier: {brier:.4f}")
p(f"  O/E: {oe:.3f} (obs={obs}, exp={exp:.1f})")
p(f"  Slope: {cal_lr.coef_[0][0]:.3f}, Intercept: {cal_lr.intercept_[0]:.3f}")
p(f"  Hosmer-Lemeshow: chi2={hl_stat:.2f}, p={hl_p:.3f}")

# ── Classification at tier boundaries ─────────────────────
p(f"\nCLASSIFICATION AT TIER BOUNDARIES:")
for name, thresh in [("Very low (<0.5%)", 0.005), ("Low (<1.5%)", 0.015), ("Moderate (<3%)", 0.03)]:
    pred_pos = yp >= thresh
    tp = ((pred_pos) & (yte.values == 1)).sum()
    fn = ((~pred_pos) & (yte.values == 1)).sum()
    fp = ((pred_pos) & (yte.values == 0)).sum()
    tn = ((~pred_pos) & (yte.values == 0)).sum()
    sens = tp/(tp+fn) if (tp+fn) > 0 else 0
    spec = tn/(tn+fp) if (tn+fp) > 0 else 0
    npv = tn/(tn+fn) if (tn+fn) > 0 else 0
    lr_neg = (1-sens)/spec if spec > 0 else 0
    n_lr = tn + fn
    ibi_rate = fn/n_lr if n_lr > 0 else 0
    p(f"\n  {name} (threshold {thresh:.1%}):")
    p(f"    Sens={sens:.3f} Spec={spec:.3f} NPV={npv:.4f} LR-={lr_neg:.3f}")
    p(f"    Below threshold: {n_lr} ({100*n_lr/len(yte):.1f}%), IBI={fn}, rate={ibi_rate:.2%}")

# Four-tier summary
p(f"\n  FOUR-TIER SUMMARY:")
tiers = [
    ("Very low (P<0.5%)", yp < 0.005),
    ("Low (0.5-1.5%)", (yp >= 0.005) & (yp < 0.015)),
    ("Moderate (1.5-3%)", (yp >= 0.015) & (yp < 0.03)),
    ("High (>3%)", yp >= 0.03),
]
for name, mask in tiers:
    n_t = mask.sum()
    ibi_t = yte.values[mask].sum()
    rate = ibi_t/n_t if n_t > 0 else 0
    p(f"    {name:<25} n={n_t:>5} ({100*n_t/len(yte):>5.1f}%) IBI={ibi_t:>3} rate={rate:.2%}")

# ── Coefficients with CIs ─────────────────────────────────
p(f"\nCOEFFICIENTS:")
try:
    probs = lr.predict_proba(Xtr)
    W = np.diag(probs[:, 0] * probs[:, 1])
    H = Xtr.values.T @ W @ Xtr.values
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    for i, f in enumerate(FEATURES):
        c = lr.coef_[0][i]
        p(f"  {f:<15} {c:>8.4f} ({c-1.96*se[i]:>8.4f} to {c+1.96*se[i]:>8.4f})")
except:
    for i, f in enumerate(FEATURES):
        p(f"  {f:<15} {lr.coef_[0][i]:>8.4f}")
p(f"  {'intercept':<15} {lr.intercept_[0]:>8.4f}")

# ── Bootstrap ─────────────────────────────────────────────
p(f"\nBOOTSTRAP (200 resamples):")
X_all = pd.concat([Xtr, Xte]); y_all = pd.concat([ytr, yte])
boot_aucs = []; boot_briers = []; boot_miss_05 = []; boot_miss_15 = []
rng = np.random.default_rng(42)
for b in range(200):
    idx = rng.choice(len(X_all), size=len(X_all), replace=True)
    oob = np.setdiff1d(np.arange(len(X_all)), idx)
    if len(oob) < 10 or y_all.iloc[oob].sum() < 2: continue
    lr_b = LogisticRegression(max_iter=1000, random_state=42)
    lr_b.fit(X_all.iloc[idx], y_all.iloc[idx])
    yp_b = lr_b.predict_proba(X_all.iloc[oob])[:, 1]
    try:
        boot_aucs.append(roc_auc_score(y_all.iloc[oob], yp_b))
        boot_briers.append(brier_score_loss(y_all.iloc[oob], yp_b))
        boot_miss_05.append(((yp_b < 0.005) & (y_all.iloc[oob].values == 1)).sum())
        boot_miss_15.append(((yp_b < 0.015) & (y_all.iloc[oob].values == 1)).sum())
    except: continue

ba = np.array(boot_aucs)
p(f"  AUC: {ba.mean():.3f} (95% CI: {np.percentile(ba,2.5):.3f}-{np.percentile(ba,97.5):.3f})")
p(f"  Brier: {np.mean(boot_briers):.4f} (95% CI: {np.percentile(boot_briers,2.5):.4f}-{np.percentile(boot_briers,97.5):.4f})")
bm05 = np.array(boot_miss_05); bm15 = np.array(boot_miss_15)
p(f"  IBI in 'very low' (P<0.5%): mean={bm05.mean():.1f}, 0-miss={100*(bm05==0).mean():.0f}%")
p(f"  IBI in 'low' (P<1.5%): mean={bm15.mean():.1f}, range=[{bm15.min()},{bm15.max()}]")

# ── Decision curve ────────────────────────────────────────
p(f"\nDECISION CURVE:")
p(f"  {'Thresh':>8} {'Model NB':>10} {'All-treat':>10} {'Better':>8}")
for pt in [0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05]:
    pp = yp >= pt
    tp = ((pp) & (yte.values==1)).sum(); fp = ((pp) & (yte.values==0)).sum(); n = len(yte)
    nb = tp/n - fp/n*(pt/(1-pt))
    nb_all = yte.sum()/n - (yte==0).sum()/n*(pt/(1-pt))
    p(f"  {pt:>7.1%} {nb:>10.4f} {nb_all:>10.4f} {'YES' if nb > nb_all else 'no':>8}")

# ── Age-stratified performance (sensitivity analysis) ─────
p(f"\nAGE-STRATIFIED PERFORMANCE (sensitivity analysis):")
for label, lo, hi in [("0-28d", 0, 28), ("29-60d", 29, 60)]:
    mask = (complete["age_days"] >= lo) & (complete["age_days"] <= hi)
    sub_y = y[mask]
    sub_x = complete[mask][FEATURES]
    sub_pred = lr.predict_proba(sub_x)[:, 1]
    try:
        sub_auc = roc_auc_score(sub_y, sub_pred)
    except:
        sub_auc = float('nan')
    sub_lr = (sub_pred < 0.015).sum()
    sub_miss = sub_y.values[sub_pred < 0.015].sum()
    p(f"  {label}: n={len(sub_y)}, IBI={sub_y.sum()}, AUC={sub_auc:.3f}")
    p(f"    'Low' (P<1.5%): {sub_lr} ({100*sub_lr/len(sub_y):.1f}%), IBI={sub_miss}")

p(f"\n{'='*70}")
p("VALIDATION COMPLETE")
p(f"{'='*70}")
out.close()
print("Results written to results/v6_validation_output.txt")
