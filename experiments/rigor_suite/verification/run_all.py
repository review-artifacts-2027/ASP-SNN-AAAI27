"""Run formal checks and diagnostics; exit nonzero only on formal failures."""
import os, subprocess, sys
here = os.path.dirname(os.path.abspath(__file__))
scripts = sorted(f for f in os.listdir(here) if f.startswith("V") and f.endswith(".py"))
fails = []
for s in scripts:
    print(f"\n========== {s} ==========")
    r = subprocess.run([sys.executable, os.path.join(here, s)])
    if r.returncode != 0:
        fails.append(s)
print("\n==== VERIFICATION SUMMARY ====")
print("ALL FORMAL CHECKS PASS" if not fails else f"FORMAL CHECK FAILURES: {fails}")
sys.exit(1 if fails else 0)
