"""
Stage 3 — Automated comparison of data-driven model structures, plus SHAP.

Compares five model families under identical 10-fold CV, on two feature sets:
  "all"     : every logged variable, including the T_soll proxy and noise
  "physics" : drivers only, T_ist instead of T_soll, noise removed

The second set encodes chemical-engineering knowledge. The comparison asks
whether that knowledge costs predictive accuracy (it should not) and what it
buys in interpretability and transferability.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

PHYSICS = ["T_ist_C", "t_reaktion_min", "p_H2_bar", "WHSV_h",
           "Pt_beladung_wt", "ruehrer_rpm", "dbt_reinheit_pct"]


def models():
    return {
        "Ridge (linear)": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 40))),
        "Random forest": RandomForestRegressor(n_estimators=600, random_state=0, n_jobs=-1),
        "Gradient boosting": XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=3,
                                          subsample=0.8, colsample_bytree=0.8,
                                          random_state=0, n_jobs=-1),
        "Gaussian process": make_pipeline(
            StandardScaler(),
            GaussianProcessRegressor(
                kernel=ConstantKernel(1.0) * Matern(length_scale=np.ones(1), nu=2.5)
                + WhiteKernel(1.0),
                normalize_y=True, random_state=0, n_restarts_optimizer=2)),
    }


def compare(X, y, label, cv=KFold(10, shuffle=True, random_state=0)):
    rows = []
    for name, mod in models().items():
        pred = cross_val_predict(mod, X, y, cv=cv, n_jobs=1)
        rows.append({
            "feature set": label,
            "model": name,
            "R2_cv": r2_score(y, pred),
            "RMSE_cv": root_mean_squared_error(y, pred),
            "MAE_cv": mean_absolute_error(y, pred),
        })
    return rows


if __name__ == "__main__":
    df = pd.read_csv("lohc_runs.csv")
    y = df["ausbeute_pct"].to_numpy()
    X_all = df.drop(columns="ausbeute_pct")
    X_phys = df[PHYSICS]

    rows = compare(X_all, y, "all features") + compare(X_phys, y, "physics-informed")
    res = pd.DataFrame(rows)

    print("=== Automated model comparison (10-fold CV, Ausbeute in %) ===\n")
    piv = res.pivot(index="model", columns="feature set", values="R2_cv").round(4)
    piv.columns = [f"R2 [{c}]" for c in piv.columns]
    rm = res.pivot(index="model", columns="feature set", values="RMSE_cv").round(2)
    rm.columns = [f"RMSE [{c}]" for c in rm.columns]
    print(pd.concat([piv, rm], axis=1).to_string())

    res.to_csv("model_comparison.csv", index=False)
    best = res.sort_values("R2_cv", ascending=False).iloc[0]
    print(f"\nBest overall: {best['model']} on '{best['feature set']}' "
          f"(R2 = {best['R2_cv']:.4f}, RMSE = {best['RMSE_cv']:.2f} %)")
