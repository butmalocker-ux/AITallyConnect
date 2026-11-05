"""Mutation test: plant one bug at a time in ti_engine.py and confirm the test suite fails.
Usage: python3 mutation_test.py   (run from the folder holding ti_engine.py, ti_synth.py, tests/)
A mutant that SURVIVES means the tests do not protect that rule."""
import os, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
MUTANTS = [
 ("movement sign flip", "return net_debit + net_credit", "return net_credit - net_debit"),
 ("tier uses cumulative AFTER", "before = ((cum - t[\"revenue\"]) / total).round(9)", "before = (cum / total).round(9)"),
 ("ageing oldest-first", "g.sort_values(\"bill_date\", ascending=False)", "g.sort_values(\"bill_date\", ascending=True)"),
 ("bucket boundary 30->31", "[30, 60, 90, 180]", "[31, 60, 90, 180]"),
 ("fmt_inr_short rollover removed", "if round(a / 1e5, 2) >= 100:", "if False:"),
 ("full_fys includes part year", "if e > as_of:", "if e >= as_of + __import__('datetime').timedelta(days=400):"),
 ("gate any instead of all", "all(c[\"pass\"] for c in checks)", "any(c[\"pass\"] for c in checks)"),
 ("gate passes on no checks", "checks and all(", "all("),
 ("dso allows zero revenue", "revenue_ttm <= 0 else", "revenue_ttm < 0 else"),
 ("winback best=sum", "best = fyrev[cols].max(axis=1)", "best = fyrev[cols].sum(axis=1)"),
 ("lfl 29 Feb -> day 1", "day=28)", "day=1)"),
 ("white_space sum not mean", ".groupby(level=0).mean()", ".groupby(level=0).sum()"),
 ("retention no floor", "cur_fl = cur.clip(lower=0.0)", "cur_fl = cur"),
 ("tolerance 0.01 -> 1.0", "TOL = 0.01", "TOL = 1.0"),
 ("spikes factor 1.5 -> 0.5", "factor: float = 1.5", "factor: float = 0.5"),
 ("cash_released no floor", "max(revenue_ttm, 0.0) / 365", "revenue_ttm / 365"),
 ("tie ignores sign", "abs(var) <= tol", "var <= tol"),
 ("top_n uses all rows incl negatives", "pos = totals[totals > 0].sort_values", "pos = totals.sort_values"),
 ("winback cutoff 15 -> 12 months", "months: int = 15", "months: int = 12"),
 ("duplicate window 7 -> 70 days", "days: int = 7", "days: int = 70"),
]

def main():
    survived, killed = [], []
    for name, old, new in MUTANTS:
        d = tempfile.mkdtemp(prefix="mut_")
        shutil.copytree(os.path.join(HERE, "tests"), os.path.join(d, "tests"))
        for f in ("ti_engine.py", "ti_synth.py"):
            shutil.copy(os.path.join(HERE, f), d)
        p = os.path.join(d, "ti_engine.py")
        src = open(p).read()
        if old not in src:
            print(f"SKIP   {name}: pattern not found"); continue
        open(p, "w").write(src.replace(old, new, 1))
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "--color=no", "tests"],
                           cwd=d, capture_output=True, text=True, timeout=180)
        (killed if r.returncode != 0 else survived).append(name)
        print(("KILLED " if r.returncode != 0 else "SURVIVED"), name)
        shutil.rmtree(d, ignore_errors=True)
    n = len(killed) + len(survived)
    print(f"\nkill rate: {len(killed)}/{n} = {len(killed)/max(n,1):.0%}")
    if survived:
        print("SURVIVORS (tests need strengthening):", survived); sys.exit(1)

if __name__ == "__main__":
    main()
