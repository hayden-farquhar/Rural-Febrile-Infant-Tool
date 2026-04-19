"""Missed IBI case analysis and multi-threshold evaluation.

Identifies the clinical profiles of IBI cases classified as low-risk,
tests cross-validation stability of miss rates, and evaluates
conservative two-threshold approaches.

Inputs:  PECARN Biosignatures CSV files (see data/raw/README.md)
Outputs: results/threshold_and_misses_output.txt
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

os.makedirs("results", exist_ok=True)
out = open("results/threshold_and_misses_output.txt", "w")
def p(s=""): out.write(s + "\n"); out.flush()

from src.prediction_model import load_and_prepare_pecarn

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
older = complete[(complete["age_days"] > 28) & (complete["age_days"] <= 60)].copy()
y_older = older["has_ibi"].astype(int)
feats = ["temp_c", "wbc", "anc", "ua_pos", "yos_total"]

Xtr, Xte, ytr, yte = train_test_split(older, y_older, test_size=0.30, random_state=42, stratify=y_older)

# Best model: Ridge C=0.1
lr = LogisticRegression(max_iter=1000, random_state=42, penalty="l2", C=0.1)
lr.fit(Xtr[feats], ytr)
yp = lr.predict_proba(Xte[feats])[:, 1]

# ═══════════════════════════════════════════════════════════════
p("=" * 70)
p("1. MULTI-THRESHOLD ANALYSIS (Older 29-60d, Ridge C=0.1)")
p("=" * 70)

p(f"\n{'Threshold':>10} {'Sens':>6} {'Spec':>6} {'NPV':>7} {'%LR':>6} {'Miss':>5} {'IBI rate in LR':>15}")
p("-" * 60)
for t in [0.002, 0.003, 0.005, 0.007, 0.008, 0.009, 0.01, 0.012, 0.015, 0.02, 0.03]:
    pred_pos = yp >= t
    tp = ((pred_pos) & (yte.values == 1)).sum()
    fn = ((~pred_pos) & (yte.values == 1)).sum()
    fp = ((pred_pos) & (yte.values == 0)).sum()
    tn = ((~pred_pos) & (yte.values == 0)).sum()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    lr_n = (tn + fn)
    ibi_rate = fn / lr_n if lr_n > 0 else 0
    p(f"{t:>9.1%} {sens:>6.1%} {spec:>6.1%} {npv:>6.1%} {100*lr_n/len(yte):>5.1%} {fn:>5} {ibi_rate:>14.2%}")

# ═══════════════════════════════════════════════════════════════
p("\n" + "=" * 70)
p("2. WHO ARE THE MISSED IBI CASES?")
p("=" * 70)

# At 1% threshold
missed_mask = (yp < 0.01) & (yte.values == 1)
missed = Xte[missed_mask].copy()
missed["predicted_p"] = yp[missed_mask]
missed["has_ibi"] = 1

p(f"\nMissed at P<1% (n={missed_mask.sum()}):")
p(f"{'PId':>8} {'age':>5} {'temp':>6} {'WBC':>6} {'ANC':>6} {'UA':>4} {'YOS':>5} {'P(IBI)':>8}")
p("-" * 52)
for _, row in missed.iterrows():
    p(f"{int(row['PId']):>8} {row['age_days']:>5.0f} {row['temp_c']:>6.1f} {row['wbc']:>6.1f} {row['anc']:>6.1f} {row['ua_pos']:>4.0f} {row['yos_total']:>5.0f} {row['predicted_p']:>7.3%}")

# Check what type of IBI these are (blood vs CSF)
blood = pd.read_csv(f"{base_dir}/culturereview_blood.csv")
csf = pd.read_csv(f"{base_dir}/culturereview_csf.csv")
bact_pids = set(blood[blood.BloodDCCAssess == 1]["PId"])
mening_pids = set(csf[csf.CSFDCCAssess == 1]["PId"])

p(f"\nMissed case types:")
for _, row in missed.iterrows():
    pid = int(row["PId"])
    has_bact = pid in bact_pids
    has_mening = pid in mening_pids
    ibi_type = []
    if has_bact: ibi_type.append("bacteremia")
    if has_mening: ibi_type.append("meningitis")
    p(f"  PId {pid}: {', '.join(ibi_type) if ibi_type else 'unknown'}")

# ═══════════════════════════════════════════════════════════════
p("\n" + "=" * 70)
p("3. CROSS-VALIDATION STABILITY")
p("=" * 70)
p("Testing 20 different random splits to check if 3 misses is stable or a fluke:")

miss_counts = []
sens_list = []
for seed in range(20):
    Xtr_s, Xte_s, ytr_s, yte_s = train_test_split(
        older, y_older, test_size=0.30, random_state=seed, stratify=y_older)
    lr_s = LogisticRegression(max_iter=1000, random_state=42, penalty="l2", C=0.1)
    lr_s.fit(Xtr_s[feats], ytr_s)
    yp_s = lr_s.predict_proba(Xte_s[feats])[:, 1]
    missed = ((yp_s < 0.01) & (yte_s.values == 1)).sum()
    n_ibi = yte_s.sum()
    sens = 1 - missed / n_ibi if n_ibi > 0 else 1
    miss_counts.append(missed)
    sens_list.append(sens)

p(f"\nAcross 20 random splits:")
p(f"  IBI missed at P<1%: mean={np.mean(miss_counts):.1f}, range=[{min(miss_counts)}, {max(miss_counts)}]")
p(f"  Sensitivity: mean={np.mean(sens_list):.1%}, range=[{min(sens_list):.1%}, {max(sens_list):.1%}]")
p(f"  Distribution of misses: {dict(zip(*np.unique(miss_counts, return_counts=True)))}")

# ═══════════════════════════════════════════════════════════════
p("\n" + "=" * 70)
p("4. CONSERVATIVE TWO-THRESHOLD APPROACH")
p("=" * 70)
p("\nInstead of binary low-risk/not-low-risk, use three zones:")

for lo_thresh, hi_thresh in [(0.005, 0.02), (0.005, 0.03), (0.007, 0.02), (0.008, 0.02)]:
    low_risk = yp < lo_thresh
    workup = yp >= hi_thresh
    uncertain = (~low_risk) & (~workup)

    miss_lr = yte.values[low_risk].sum()
    ibi_lr = miss_lr
    n_lr = low_risk.sum()
    n_unc = uncertain.sum()
    n_wu = workup.sum()
    ibi_wu = yte.values[workup].sum()
    ibi_unc = yte.values[uncertain].sum()

    p(f"\n  Thresholds: P<{lo_thresh:.1%} → low-risk, P>{hi_thresh:.1%} → workup")
    p(f"    Low-risk: {n_lr} ({100*n_lr/len(yte):.1f}%), IBI={ibi_lr}")
    p(f"    Uncertain: {n_unc} ({100*n_unc/len(yte):.1f}%), IBI={ibi_unc}")
    p(f"    Workup: {n_wu} ({100*n_wu/len(yte):.1f}%), IBI={ibi_wu}")

p("\n" + "=" * 70)
p("DONE")
p("=" * 70)
out.close()
print("Results written to results/threshold_and_misses_output.txt")
