"""Stage 3b - SHAP explanation of the gradient-boosting model."""
import numpy as np, pandas as pd, shap, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from model_comparison import PHYSICS

df = pd.read_csv("lohc_runs.csv")
y = df["ausbeute_pct"].to_numpy(); X = df[PHYSICS]

mod = XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=3,
                   subsample=0.8, colsample_bytree=0.8, random_state=0, n_jobs=-1).fit(X, y)
sv = shap.TreeExplainer(mod).shap_values(X)

imp = pd.Series(np.abs(sv).mean(0), index=PHYSICS).sort_values(ascending=False)
print("Mean |SHAP| (percentage points of yield):")
print(imp.round(2).to_string())

# Does SHAP recover the 600 rpm mass-transfer threshold?
i = PHYSICS.index("ruehrer_rpm")
rpm, s = X["ruehrer_rpm"].to_numpy(), sv[:, i]
lo, hi = rpm < 600, rpm >= 600
print(f"\nruehrer_rpm SHAP,  < 600 rpm: mean {s[lo].mean():+.2f} pp  (n={lo.sum()})")
print(f"ruehrer_rpm SHAP, >= 600 rpm: mean {s[hi].mean():+.2f} pp  (n={hi.sum()})")
print(f"slope  < 600 rpm: {np.polyfit(rpm[lo], s[lo], 1)[0]*100:+.3f} pp per 100 rpm")
print(f"slope >= 600 rpm: {np.polyfit(rpm[hi], s[hi], 1)[0]*100:+.3f} pp per 100 rpm")

fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
imp.iloc[::-1].plot.barh(ax=ax[0], color="#1D9E75")
ax[0].set_xlabel("mean |SHAP|  [Prozentpunkte Ausbeute]"); ax[0].set_title("Merkmalswichtigkeit (SHAP)")
ax[1].scatter(rpm, s, s=14, alpha=.6, c=X["T_ist_C"], cmap="coolwarm")
ax[1].axvline(600, ls="--", c="crimson"); ax[1].axhline(0, lw=.6, c="gray")
ax[1].set_xlabel("Rührerdrehzahl [rpm]"); ax[1].set_ylabel("SHAP-Beitrag [pp]")
ax[1].set_title("Stofftransport-Grenze bei 600 rpm?")
plt.tight_layout(); plt.savefig("shap_summary.png", dpi=115)
