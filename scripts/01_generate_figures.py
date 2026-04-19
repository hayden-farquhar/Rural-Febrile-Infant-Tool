"""Generate publication figures (flow diagram, calibration, decision curve, risk tiers, degradation).

Inputs:  PECARN Biosignatures CSV files (see data/raw/README.md)
Outputs: outputs/figures/figure1_flow.png through figure5_degradation.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings("ignore")

from src.prediction_model import load_and_prepare_pecarn, FEATURES

figdir = "outputs/figures"
os.makedirs(figdir, exist_ok=True)

# Load and prepare data
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

Xtr, Xte, ytr, yte = train_test_split(complete[FEATURES], y, test_size=0.30, random_state=42, stratify=y)
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(Xtr, ytr)
yp = lr.predict_proba(Xte)[:, 1]

# ── Figure 2: Calibration plot ────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=300)
prob_true, prob_pred = calibration_curve(yte, yp, n_bins=8, strategy="quantile")
ax.plot([0, 0.15], [0, 0.15], "k--", linewidth=0.8, label="Ideal calibration")
ax.plot(prob_pred, prob_true, "o-", color="#2166ac", markersize=8, linewidth=2, label="Observed")
ax.set_xlabel("Predicted IBI probability", fontsize=12)
ax.set_ylabel("Observed IBI proportion", fontsize=12)
ax.set_xlim(0, 0.12)
ax.set_ylim(0, 0.12)
ax.set_aspect("equal")
ax.legend(fontsize=10, loc="upper left")
ax.set_title("Calibration plot", fontsize=13, fontweight="bold")
# Add rug plot
ax.scatter(yp[yte.values == 0], np.full(sum(yte.values == 0), -0.002),
           marker="|", color="#999999", alpha=0.3, s=20, label="_nolegend_")
ax.scatter(yp[yte.values == 1], np.full(sum(yte.values == 1), -0.004),
           marker="|", color="#b2182b", alpha=0.8, s=40, label="_nolegend_")
ax.text(0.08, 0.005, f"Brier = {brier_score_loss(yte, yp):.4f}\nSlope = 0.937\nO/E = 1.012",
        fontsize=9, ha="left", va="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
plt.tight_layout()
plt.savefig(f"{figdir}/figure2_calibration.png", dpi=300, bbox_inches="tight")
plt.close()
print("Figure 2: calibration plot saved")

# ── Figure 3: Decision curve ─────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(7, 5), dpi=300)
thresholds = np.arange(0.002, 0.08, 0.001)
nb_model = []
nb_all = []
nb_none = []
for pt in thresholds:
    pred_pos = yp >= pt
    tp = ((pred_pos) & (yte.values == 1)).sum()
    fp = ((pred_pos) & (yte.values == 0)).sum()
    n = len(yte)
    nb_model.append(tp / n - fp / n * (pt / (1 - pt)))
    nb_all.append(yte.sum() / n - (yte == 0).sum() / n * (pt / (1 - pt)))
    nb_none.append(0)

ax.plot(thresholds * 100, nb_model, color="#2166ac", linewidth=2, label="Prediction model")
ax.plot(thresholds * 100, nb_all, color="#b2182b", linewidth=1.5, linestyle="--", label="Treat all")
ax.plot(thresholds * 100, nb_none, color="#999999", linewidth=1, linestyle=":", label="Treat none")
ax.set_xlabel("Threshold probability (%)", fontsize=12)
ax.set_ylabel("Net benefit", fontsize=12)
ax.set_xlim(0, 6)
ax.set_ylim(-0.02, 0.025)
ax.legend(fontsize=10)
ax.axhline(0, color="black", linewidth=0.3)
ax.set_title("Decision curve analysis", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{figdir}/figure3_decision_curve.png", dpi=300, bbox_inches="tight")
plt.close()
print("Figure 3: decision curve saved")

# ── Figure 4: Four-tier distribution ──────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(7, 5), dpi=300)
tiers = ["Very low\n(<0.5%)", "Low\n(0.5-1.5%)", "Moderate\n(1.5-3%)", "High\n(>3%)"]
counts = [194, 689, 244, 204]
ibi_counts = [0, 6, 7, 13]
colors = ["#4daf4a", "#377eb8", "#ff7f00", "#e41a1c"]
bars = ax.bar(tiers, counts, color=colors, edgecolor="black", linewidth=0.5, alpha=0.8)
# Add IBI counts on top
for i, (bar, ibi) in enumerate(zip(bars, ibi_counts)):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
            f"IBI: {ibi}" + (f"\n({100*ibi/counts[i]:.1f}%)" if counts[i] > 0 else ""),
            ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylabel("Number of infants", fontsize=12)
ax.set_title("Four-tier risk stratification (test set, n=1,331)", fontsize=13, fontweight="bold")
ax.set_ylim(0, 800)
plt.tight_layout()
plt.savefig(f"{figdir}/figure4_risk_tiers.png", dpi=300, bbox_inches="tight")
plt.close()
print("Figure 4: risk tier distribution saved")

# ── Figure 5: Missing-input degradation ───────────────────────
fig, ax = plt.subplots(1, 1, figsize=(7, 5), dpi=300)
features_removed = ["None\n(full)", "PCT", "Age", "ANC", "UA", "Temp", "WBC"]
aucs = [0.782, 0.686, 0.756, 0.779, 0.787, 0.781, 0.789]
colors_bar = ["#2166ac"] + ["#b2182b" if a < 0.75 else "#ff7f00" if a < 0.78 else "#4daf4a" for a in aucs[1:]]
bars = ax.barh(features_removed, aucs, color=colors_bar, edgecolor="black", linewidth=0.5, alpha=0.8)
ax.set_xlabel("AUC", fontsize=12)
ax.set_xlim(0.65, 0.80)
ax.axvline(0.782, color="#2166ac", linewidth=1, linestyle="--", alpha=0.5)
ax.set_title("Missing-input degradation\n(feature removed, PCT-available subset)", fontsize=13, fontweight="bold")
for i, (bar, auc) in enumerate(zip(bars, aucs)):
    delta = auc - 0.782
    label = f"{auc:.3f}" if i == 0 else f"{auc:.3f} ({delta:+.3f})"
    ax.text(auc + 0.002, bar.get_y() + bar.get_height() / 2, label,
            va="center", fontsize=9)
plt.tight_layout()
plt.savefig(f"{figdir}/figure5_degradation.png", dpi=300, bbox_inches="tight")
plt.close()
print("Figure 5: degradation plot saved")

# ── eFigure 1: PRISMA flow ───────────────────────────────────
# Simple text-based PRISMA flow (will need manual formatting for publication)
fig, ax = plt.subplots(1, 1, figsize=(8, 10), dpi=300)
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis("off")

boxes = [
    (5, 13, "Records identified through\ntargeted literature search\n(n = 15)"),
    (5, 11, "Full-text articles assessed\nfor eligibility\n(n = 12)"),
    (5, 9, "Studies included in\nmeta-analysis\n(n = 6, yielding 11\ncohort-rule combinations)"),
    (5, 6.5, "PECARN Biosignatures\npublic-use dataset\n(n = 6,009 infants)\nAccessed for prediction model"),
    (5, 4, "Complete cases for\nmodel development\n(n = 4,434; 88 IBI)"),
]
for x, y_pos, text in boxes:
    ax.text(x, y_pos, text, ha="center", va="center", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="black"))

# Exclusion boxes
excl = [
    (8.5, 11, "Excluded (n = 6):\n- Case-control design (1)\n- No extractable 2×2 (2)\n- Guideline only (1)\n- Biomarker-only data (2)"),
    (8.5, 4, "Incomplete cases (n = 1,575)\nIBI rate 0.51%\n(informative missingness)"),
]
for x, y_pos, text in excl:
    ax.text(x, y_pos, text, ha="center", va="center", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", edgecolor="gray"))

# Arrows
for y_start, y_end in [(12.5, 11.8), (10.2, 9.8), (8.2, 7.3), (5.8, 4.8)]:
    ax.annotate("", xy=(5, y_end), xytext=(5, y_start),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
# Exclusion arrows
ax.annotate("", xy=(7.5, 11), xytext=(6.5, 11),
            arrowprops=dict(arrowstyle="->", color="gray", lw=1))
ax.annotate("", xy=(7.5, 4), xytext=(6.5, 4),
            arrowprops=dict(arrowstyle="->", color="gray", lw=1))

ax.set_title("Figure 1: Study flow diagram", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{figdir}/figure1_flow.png", dpi=300, bbox_inches="tight")
plt.close()
print("Figure 1: flow diagram saved")

print(f"\nAll figures saved to {figdir}/")
