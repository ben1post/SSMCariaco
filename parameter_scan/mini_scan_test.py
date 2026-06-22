"""
mini_scan_test.py -- serial vs spawn-Pool on the minimal chemostat.

Run:  python mini_scan_test.py

Verifies, on a transparent model with a known parameter set, that:
  - one run_point is fast and deterministic,
  - the Pool does NOT hang (worker is imported from mini_xso_model, not here),
  - the speedup is ~processes-fold.

If 'serial' is too quick to be meaningful, lower MAX_STEP (more RHS evals per
task). If you want to mimic many combos, raise N.
"""
import time
import multiprocessing as mp

from mini_xso_model import run_point   # imported -> spawn-safe

PROCS = 6
N = 2 * PROCS            # ~2 tasks/worker, shows model-reuse amortisation
MAX_STEP = 0.02          # cost dial; smaller = heavier per task

if __name__ == "__main__":
    ctx = mp.get_context("spawn")        # mirror macOS default explicitly
    mus = [0.6 + 0.05 * i for i in range(N)]
    args = [(mu, 400.0, 0.1, MAX_STEP) for mu in mus]

    # sanity: one run, deterministic
    t0 = time.perf_counter()
    r = run_point(*args[0])
    print(f"one run_point: {r}  ({time.perf_counter() - t0:.3f}s)")

    t0 = time.perf_counter()
    serial = [run_point(*a) for a in args]
    ts = time.perf_counter() - t0

    t0 = time.perf_counter()
    with ctx.Pool(PROCS) as p:
        pooled = p.starmap(run_point, args)
    tp = time.perf_counter() - t0

    # identical results, order preserved by starmap
    ok = all(abs(a["P_final"] - b["P_final"]) < 1e-9 for a, b in zip(serial, pooled))
    print(f"serial {ts:.2f}s | pool {tp:.2f}s | speedup {ts / tp:.1f}x "
          f"(procs={PROCS}) | results match: {ok}")