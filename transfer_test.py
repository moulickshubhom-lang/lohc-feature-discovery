"""
Stage 4 - Transfer test: does the choice of temperature variable matter
when the reactor changes?

Models are trained on the original campaign (lohc_runs.csv) and evaluated
on a simulated scale-up reactor in which the controller-setpoint to
measured-temperature relationship has shifted. The kinetics are unchanged.
"""
import numpy as np, pandas as pd
from sklearn.metrics import r2_score, root_mean_squared_error
from generate_data import generate
from model_comparison import models, PHYSICS

DRIVERS = [f for f in PHYSICS if f != "T_ist_C"]
SETS = {
    "proxy   (T_soll + drivers)":   ["T_soll_C"] + DRIVERS,
    "physics (T_ist  + drivers)":   ["T_ist_C"]  + DRIVERS,
    "all     (both T + noise)":     None,
}

train = pd.read_csv("lohc_runs.csv")
scenarios = {
    "same reactor (sanity)":         dict(endo_scale=1.0, endo_bias=0.0),
    "scale-up: dip x2":              dict(endo_scale=2.0, endo_bias=0.0),
    "scale-up: dip x2 + 8 K offset": dict(endo_scale=2.0, endo_bias=8.0),
}

rows = []
for mname in ["Gradient boosting", "Gaussian process"]:
    for sname, cols in SETS.items():
        Xtr = train.drop(columns="ausbeute_pct") if cols is None else train[cols]
        mod = models()[mname].fit(Xtr, train.ausbeute_pct)
        for scen, kw in scenarios.items():
            test = generate(n=200, rng=np.random.default_rng(99), **kw)
            Xte = test.drop(columns="ausbeute_pct") if cols is None else test[cols]
            p = mod.predict(Xte)
            rows.append(dict(model=mname, features=sname, scenario=scen,
                             R2=r2_score(test.ausbeute_pct, p),
                             RMSE=root_mean_squared_error(test.ausbeute_pct, p)))

res = pd.DataFrame(rows)
res.to_csv("transfer_results.csv", index=False)
pd.set_option("display.width", 160)
for m in res.model.unique():
    print(f"\n=== {m} ===")
    sub = res[res.model == m]
    print(sub.pivot(index="features", columns="scenario", values="RMSE").round(2)
             .loc[list(SETS)][list(scenarios)].to_string())
