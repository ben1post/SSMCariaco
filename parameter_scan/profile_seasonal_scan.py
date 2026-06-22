"""
profile_seasonal_scan.py -- measure WHERE run_one spends time and whether a
process Pool buys the expected speedup, BEFORE committing to a parallelisation
design. Times only; writes nothing to disk.

PLACE THIS NEXT TO seasonal_scan_harness.py (or set HARNESS_DIR below), then:

    python profile_seasonal_scan.py             # pinned BLAS (default)
    PIN_BLAS=0 python profile_seasonal_scan.py  # unpinned, for the A/B

It answers three questions with numbers instead of assertions:

  [1] one combo, split setup-build / solve / reduce
        -> is per-task xso.setup negligible? If yes, run_one-as-worker is the
           clean design; if not, weigh init-once + update_vars-delta.
  [2] numpy's BLAS thread pool
        -> is oversubscription even possible at this system size?
  [3] serial N combos vs Pool(N)
        -> the real speedup; run twice (PIN_BLAS=1 vs 0) to see whether
           thread-pinning actually matters here.

run_one is used UNCHANGED as the worker (module-level + picklable), so the Pool
part is macOS-spawn-safe. Nothing in seasonal_scan_harness.py is modified.
"""
import os

# --- BLAS/OMP pinning MUST be set before numpy is imported -- here in the parent,
#     and in spawned workers which re-import this module + the harness. Toggle via
#     the PIN_BLAS env var so the [3] A/B is two clean runs. ----------------------
PIN_BLAS = os.environ.get("PIN_BLAS", "1") == "1"
if PIN_BLAS:
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(_v, "1")

HARNESS_DIR = None  # set to the folder holding seasonal_scan_harness.py if not CWD
if HARNESS_DIR:
    import sys
    sys.path.insert(0, HARNESS_DIR)

import time
from multiprocessing import Pool

import numpy as np

try:
    import seasonal_scan_harness as ssh
except ImportError as e:
    raise SystemExit(
        f"Could not import seasonal_scan_harness ({e}).\n"
        f"Run this from the folder containing it, or set HARNESS_DIR at the top."
    )

# ------------------------------- knobs -------------------------------------
PROFILE_YEARS  = 60   # use the length you ACTUALLY run; the setup/solve ratio
PROFILE_SPINUP = 15   #   is what decides the design (lower for a quick look)
PARALLEL_YEARS  = 15  # shorter is fine -- speedup ratio is ~years-independent
PARALLEL_SPINUP = 3   #   (kept < PARALLEL_YEARS so _clim has a valid window)
N_COMBOS  = max(2, 2 * (os.cpu_count() or 2) - 2)  # ~2 per worker -> shows amortisation
PROCESSES = max(1, (os.cpu_count() or 2) - 1)
# ---------------------------------------------------------------------------


def profile_one_combo():
    """Replicate run_one's body, timing setup-build / solve / reduce separately."""
    construct = ssh.allometry("taniguchi")
    forcings = ssh.build_forcings()
    group = list(forcings)[0]
    forcing = forcings[group]
    print(f"\n[1] single-combo profile  (construct=taniguchi, group={group}, "
          f"years={PROFILE_YEARS})")

    clk = time.perf_counter

    t0 = clk()  # --- setup build (make_seasonal_input_vars + xso.setup) ---
    iv = ssh.make_seasonal_input_vars(
        forcing["fn"], forcing["de"], forcing["t"], fish_rate=0.0,
        mu_max=construct["mu_max"], halfsat=construct["halfsat"],
        mP=ssh.M_P, m_Z=ssh.M_Z_BULK, spline_s=0.0)
    time_ax = np.arange(0.0, PROFILE_YEARS * 365.0 + 1.0, 1.0)
    setup = ssh.xso.setup(
        solver="solve_ivp", model=ssh.model_baseline_seasonal, time=time_ax,
        input_vars=iv, output_vars=ssh.SLIM_OUTPUT_VARS,
        solver_kwargs=ssh.SEASONAL_SOLVER_KWARGS)
    t_setup = clk() - t0

    t0 = clk()  # --- solve (the IVP; expected to dominate) ---
    out = ssh.pue.run_single_point(ssh.model_baseline_seasonal, setup, {})
    t_solve = clk() - t0

    t0 = clk()  # --- reduce (_reduce + Export areal + _clim), exactly as run_one ---
    r = ssh._reduce(out)
    if "Export" in r:
        de_t = ssh._build_fn_func(forcing["de"], ssh.PERIOD, ssh.SPLINE_K, 0.0)(r["t"])
        r["Export"] = r["Export"] * de_t
    _ = ssh._clim(r, PROFILE_SPINUP)
    t_reduce = clk() - t0

    total = t_setup + t_solve + t_reduce
    for name, dt in [("setup build", t_setup), ("solve_ivp", t_solve),
                     ("reduce", t_reduce)]:
        print(f"    {name:12s} {dt:8.3f}s  ({100 * dt / total:5.1f}%)")
    print(f"    {'TOTAL':12s} {total:8.3f}s")
    verdict = ("negligible -> keep run_one-as-worker" if t_setup / total < 0.02
               else "non-trivial -> weigh init-once + update_vars-delta")
    print(f"    -> setup-build = {100 * t_setup / total:.1f}% of a combo: {verdict}")


def report_blas():
    print("\n[2] BLAS thread pool")
    print(f"    PIN_BLAS={PIN_BLAS}  cpu_count={os.cpu_count()}")
    print("    env:", {v: os.environ.get(v) for v in
                       ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")})
    try:
        from threadpoolctl import threadpool_info
        for p in threadpool_info():
            print(f"    {p.get('internal_api')}: {p.get('num_threads')} threads "
                  f"({p.get('prefix')})")
    except ImportError:
        print("    (pip/conda install threadpoolctl to see numpy's real BLAS thread count)")


def _combos():
    forcings = ssh.build_forcings()
    constructs = [ssh.allometry(c) for c in ssh.DEFAULT_CONSTRUCTS]
    groups = list(forcings)
    # positional args for run_one(construct, forcing, fish_rate, years, spinup)
    return [(constructs[i % len(constructs)], forcings[groups[i % len(groups)]],
             0.0, PARALLEL_YEARS, PARALLEL_SPINUP) for i in range(N_COMBOS)]


def run_parallel_test():
    combos = _combos()
    print(f"\n[3] serial vs Pool  ({N_COMBOS} combos, years={PARALLEL_YEARS}, "
          f"processes={PROCESSES}, PIN_BLAS={PIN_BLAS})")

    t0 = time.perf_counter()
    for c in combos:
        try:
            ssh.run_one(*c)
        except Exception as e:
            print(f"    [serial warn] {e}")
    t_serial = time.perf_counter() - t0
    print(f"    serial : {t_serial:7.1f}s")

    t0 = time.perf_counter()
    with Pool(processes=PROCESSES) as p:
        p.starmap(ssh.run_one, combos)
    t_pool = time.perf_counter() - t0
    print(f"    Pool   : {t_pool:7.1f}s   speedup {t_serial / max(t_pool, 1e-9):4.1f}x "
          f"(ideal ~{PROCESSES}x)")


if __name__ == "__main__":
    profile_one_combo()
    report_blas()
    run_parallel_test()
    print("\nRe-run with  PIN_BLAS=0 python profile_seasonal_scan.py  and compare the "
          "Pool time to see whether thread-pinning matters at this system size.")