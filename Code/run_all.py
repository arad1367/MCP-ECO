import subprocess
import sys
import time

SCRIPTS = [
    "01_prepare.py",
    "02_validation.py",
    "03_reclassify.py",
    "04_features.py",
    "05_rq1_lca.py",
    "06_rq2_adoption.py",
    "07_rq3_disclosure.py",
    "08_rq4_language.py",
    "09_panel_robustness.py",
    "10_report.py",
]


def main():
    for s in SCRIPTS:
        t0 = time.time()
        print(f"\n>>> {s}", flush=True)
        r = subprocess.run([sys.executable, s])
        if r.returncode != 0:
            print(f"FAILED: {s}")
            sys.exit(r.returncode)
        print(f"<<< {s} finished in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
