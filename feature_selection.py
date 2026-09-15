"""
Stage 2 — Comparison of feature-selection methods, graded against ground truth.

Methods
  1. Pearson correlation        -- linear, univariate  (baseline)
  2. Mutual information         -- nonlinear, univariate
  3. Lasso (LassoCV)            -- linear, multivariate, sparse
  4. Random forest importance   -- nonlinear, multivariate
  5. Stability selection        -- Lasso refit on bootstrap resamples

The interesting case is the pair (T_soll, T_ist): near-collinear (r ~ 0.99),
but only T_ist is the proximate physical cause. Statistical criteria alone
cannot separate them; stability selection at least reveals the ambiguity.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from generate_data import GROUND_TRUTH

RNG = np.random.default_rng(7)


def load():
    df = pd.read_csv("lohc_runs.csv")
    y = df["ausbeute_pct"].to_numpy()
    X = df.drop(columns="ausbeute_pct")
    return X, y


def rank_table(X, y):
    names = list(X.columns)
    Xv = X.to_numpy()
    Xs = StandardScaler().fit_transform(Xv)

    pearson = np.array([abs(np.corrcoef(Xv[:, i], y)[0, 1]) for i in range(Xv.shape[1])])
    mi = mutual_info_regression(Xv, y, random_state=0)

    lasso = LassoCV(cv=10, random_state=0, max_iter=50_000).fit(Xs, y)
    lasso_coef = np.abs(lasso.coef_)

    rf = RandomForestRegressor(n_estimators=600, random_state=0, n_jobs=-1).fit(Xv, y)
    rf_imp = rf.feature_importances_

    out = pd.DataFrame({
        "pearson": pearson,
        "mutual_info": mi,
        "lasso_abs_coef": lasso_coef,
        "rf_importance": rf_imp,
    }, index=names)

    # normalise each column to 0-1 so methods are comparable side by side
    norm = out / out.max()
    norm["truth"] = [GROUND_TRUTH[n] for n in names]
    return out, norm, lasso.alpha_


def stability_selection(X, y, n_boot=400, frac=0.75, alpha=None):
    """Refit Lasso on bootstrap subsamples; report selection frequency."""
    names = list(X.columns)
    Xv = X.to_numpy()
    n = len(y)
    n_sub = int(frac * n)
    counts = np.zeros(Xv.shape[1])

    for _ in range(n_boot):
        idx = RNG.choice(n, size=n_sub, replace=False)
        Xs = StandardScaler().fit_transform(Xv[idx])
        a = alpha if alpha is not None else 0.05
        mod = Lasso(alpha=a, max_iter=50_000).fit(Xs, y[idx])
        counts += (np.abs(mod.coef_) > 1e-8)

    return pd.Series(counts / n_boot, index=names).sort_values(ascending=False)


def stability_grid(X, y, alphas=(0.15, 0.5, 1.5, 3.0, 6.0), n_boot=200):
    """Selection frequency for each feature across a grid of Lasso penalties.

    The point of the grid: with two near-collinear temperature variables, which
    one 'survives' is a function of the penalty, not of the chemistry.
    """
    return pd.DataFrame(
        {f"alpha={a}": stability_selection(X, y, n_boot=n_boot, alpha=a) * 100
         for a in alphas}
    )


def plot_stability(grid, alphas=(0.15, 0.5, 1.5, 3.0, 6.0), path="stability_selection.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for feat, style in [("T_ist_C", dict(color="#1D9E75", lw=2.4)),
                        ("T_soll_C", dict(color="#C1440E", lw=2.4, ls="--"))]:
        ax.plot([str(a) for a in alphas], grid.loc[feat], marker="o", label=feat, **style)
    ax.set_xlabel("Lasso-Strafterm alpha")
    ax.set_ylabel("Auswahlhaeufigkeit [%]")
    ax.set_title("Welche Temperatur wird ausgewaehlt? Haengt vom Strafterm ab.")
    ax.set_ylim(0, 105); ax.legend(); ax.grid(alpha=.25)
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


if __name__ == "__main__":
    X, y = load()
    raw, norm, alpha = rank_table(X, y)

    pd.set_option("display.width", 140)
    print(f"LassoCV alpha = {alpha:.4f}\n")
    print("=== Normalised importance (1.00 = top-ranked by that method) ===")
    print(norm.round(3).to_string())

    print("\n=== Lasso: features driven exactly to zero ===")
    dropped = raw.index[raw.lasso_abs_coef < 1e-8].tolist()
    print(dropped if dropped else "(none)")

    print("\n=== Stability selection across a penalty grid (selection freq., %) ===")
    print("    200 bootstrap resamples, 75% subsample each\n")
    grid = stability_grid(X, y)
    order = ["T_ist_C", "t_reaktion_min", "p_H2_bar", "WHSV_h", "Pt_beladung_wt",
             "ruehrer_rpm", "dbt_reinheit_pct", "T_soll_C", "charge_nr",
             "T_umgebung_C", "lagerzeit_tage"]
    print(grid.loc[order].round(0).to_string())
    grid.to_csv("stability_selection.csv")
    plot_stability(grid)
    print("\n-> stability_selection.png / .csv")
    print("\nNote: the causal T_ist dominates at weak penalties, the non-causal")
    print("T_soll takes over at strong ones. No data-driven rule picks between")
    print("them -- only the Arrhenius argument does.")
