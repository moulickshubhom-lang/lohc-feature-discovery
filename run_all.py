"""Reproduce every result and figure in this repository, in order."""
import subprocess, sys, time

STAGES = [
    ("generate_data.py",     "Stage 1  synthetic reactor campaign"),
    ("feature_selection.py", "Stage 2  feature-selection comparison"),
    ("model_comparison.py",  "Stage 3  model-structure comparison"),
    ("shap_analysis.py",     "Stage 3b SHAP explanation"),
    ("transfer_test.py",     "Stage 4  scale-up transfer test"),
]

if __name__ == "__main__":
    for script, label in STAGES:
        print(f"\n{'='*70}\n{label}  ({script})\n{'='*70}")
        t0 = time.time()
        r = subprocess.run([sys.executable, script])
        if r.returncode:
            sys.exit(f"FAILED: {script}")
        print(f"[{time.time()-t0:.1f} s]")
    print("\nAll stages complete.")
