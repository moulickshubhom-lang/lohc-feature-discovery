"""
Stage 1 — Synthetic LOHC dehydrogenation dataset.

Generates batch-reactor runs for H18-DBT -> H0-DBT dehydrogenation over Pt/Al2O3.
Ground truth uses Langmuir-Hinshelwood kinetics with H2 product inhibition,
consistent with the PFR model in `lohc-dehydrogenation-sim`.

Feature design is deliberate (see README):
  - true drivers        : T_ist, p_H2, WHSV, t_reaktion, Pt_beladung
  - correlated proxy    : T_soll (controller setpoint; r ~ 0.98 with T_ist,
                          logged by the autoclave but NOT the causal variable --
                          the reaction responds to the measured internal
                          temperature, not to the instruction given to the
                          controller)
  - conditionally irrel.: ruehrer_rpm (matters only below mass-transfer threshold)
  - weak driver         : dbt_reinheit
  - pure noise          : charge_nr, T_umgebung, lagerzeit_tage
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N_RUNS = 280          # matches the scale of the real Hydrogenious campaign
R = 8.314             # J/(mol K)


def generate(n=N_RUNS, rng=RNG, endo_scale=1.0, endo_bias=0.0):
    """endo_scale / endo_bias let a second reactor be simulated with a
    different setpoint-to-measured relationship (e.g. scale-up with poorer
    heat transfer) while the kinetics stay identical."""
    # ---------------------------------------------------------------
    # 1. Operating conditions (experimental design space)
    # ---------------------------------------------------------------
    # T_soll is what the operator sets; T_ist is what the reaction feels.
    T_soll = rng.uniform(255, 320, n)               # degC, controller setpoint
    p_H2 = rng.uniform(1.0, 5.0, n)                 # bar, backpressure
    WHSV = rng.uniform(0.5, 4.0, n)                 # 1/h
    t_reaktion = rng.uniform(15, 300, n)            # min
    Pt_beladung = rng.uniform(0.3, 1.0, n)          # wt% Pt
    ruehrer_rpm = rng.uniform(200, 1200, n)         # rpm
    dbt_reinheit = rng.uniform(95.0, 99.9, n)       # % H18-DBT purity

    # ---------------------------------------------------------------
    # 2. Measured internal temperature. Dehydrogenation is endothermic, so
    #    T_ist sits below setpoint; the offset grows with catalyst activity
    #    (faster reaction -> larger heat draw) and with controller jitter.
    #    Result: T_soll is a strong but imperfect, NON-CAUSAL proxy of T_ist.
    # ---------------------------------------------------------------
    endo_last = endo_bias + endo_scale * (2.0 + 6.0 * (Pt_beladung / 1.0))  # K
    T_ist = T_soll - endo_last + rng.normal(0.0, 2.2, n)

    # ---------------------------------------------------------------
    # 3. Pure noise features (no causal path to yield)
    # ---------------------------------------------------------------
    charge_nr = rng.integers(1000, 1400, n).astype(float)
    T_umgebung = rng.uniform(18, 26, n)
    lagerzeit_tage = rng.uniform(0, 180, n)

    # ---------------------------------------------------------------
    # 4. Ground-truth kinetics (Langmuir-Hinshelwood, H2 inhibition)
    # ---------------------------------------------------------------
    T_K = T_ist + 273.15
    Ea = 95_000.0                                   # J/mol
    k0 = 4.5e7
    k = k0 * np.exp(-Ea / (R * T_K))                # 1/min, intrinsic

    # Active-site scaling: sub-linear in Pt loading (dispersion saturates)
    site_faktor = (Pt_beladung / 0.5) ** 0.6

    # H2 product inhibition in the LH denominator
    K_H2 = 0.45                                     # 1/bar
    inhibition = 1.0 / (1.0 + K_H2 * p_H2) ** 2

    # Contact-time effect: high WHSV -> less residence per unit catalyst
    whsv_faktor = (1.0 / WHSV) ** 0.35

    # Mass-transfer limitation: only binds BELOW ~600 rpm. Above that the
    # batch reactor is well mixed and stirrer speed is physically irrelevant.
    mt_faktor = np.clip(ruehrer_rpm / 600.0, None, 1.0) ** 0.5

    # Purity: weak, near-linear penalty for impurities
    reinheit_faktor = 1.0 - 0.015 * (99.9 - dbt_reinheit)

    k_eff = k * site_faktor * inhibition * whsv_faktor * mt_faktor * reinheit_faktor

    # Batch conversion: first-order approach to equilibrium
    X_eq = 0.98
    umsatz = X_eq * (1.0 - np.exp(-k_eff * t_reaktion))

    # ---------------------------------------------------------------
    # 5. Measurement noise — heteroscedastic (GC-FID is less precise at
    #    low conversion), mirroring real analytical behaviour.
    # ---------------------------------------------------------------
    noise_sd = 0.008 + 0.035 * (1.0 - umsatz)
    ausbeute = umsatz + rng.normal(0.0, noise_sd, n)
    ausbeute = np.clip(ausbeute, 0.0, 1.0) * 100.0  # -> %

    df = pd.DataFrame({
        "T_soll_C": T_soll,
        "T_ist_C": T_ist,
        "p_H2_bar": p_H2,
        "WHSV_h": WHSV,
        "t_reaktion_min": t_reaktion,
        "Pt_beladung_wt": Pt_beladung,
        "ruehrer_rpm": ruehrer_rpm,
        "dbt_reinheit_pct": dbt_reinheit,
        "charge_nr": charge_nr,
        "T_umgebung_C": T_umgebung,
        "lagerzeit_tage": lagerzeit_tage,
        "ausbeute_pct": ausbeute,
    })
    return df


# Ground truth, for honest scoring of the feature-selection methods later
GROUND_TRUTH = {
    "T_ist_C": "driver (Arrhenius, dominant) -- CAUSAL",
    "t_reaktion_min": "driver (batch conversion)",
    "p_H2_bar": "driver (LH product inhibition)",
    "WHSV_h": "driver (contact time)",
    "Pt_beladung_wt": "driver (active sites, sub-linear)",
    "ruehrer_rpm": "conditional (only < 600 rpm)",
    "dbt_reinheit_pct": "weak driver",
    "T_soll_C": "PROXY: controller setpoint, NOT causal",
    "charge_nr": "noise",
    "T_umgebung_C": "noise",
    "lagerzeit_tage": "noise",
}


if __name__ == "__main__":
    df = generate()
    df.to_csv("lohc_runs.csv", index=False)
    print(f"{len(df)} runs -> lohc_runs.csv\n")
    print(df.describe().T[["mean", "std", "min", "max"]].round(2))
    print("\nAusbeute-Verteilung:")
    print(df["ausbeute_pct"].describe().round(2))
    print("\nKorrelation T_soll <-> T_ist: "
          f"{df['T_soll_C'].corr(df['T_ist_C']):.3f}")
    print("Mittlere Soll-Ist-Abweichung: "
          f"{(df['T_ist_C'] - df['T_soll_C']).mean():.2f} K")
