"""
mini_leak_repro.py -- minimal, FAST XSO model to reproduce (or rule out) the
per-cell RSS leak without the full MS3 model.

It keeps the one leak-relevant feature and drops everything else: a vector state
of N classes whose flux allocates an N x N temporary EVERY RHS call -- the cheap
analogue of the real model's matrix-valued grazing flux, i.e. the per-call
allocation churn the fragmentation hypothesis says drives the leak. Dynamics are
stable logistic competition, so it integrates to the end (no NaN-termination)
and is deterministic.

At N=80, T=14601 a cell is < 1 s (vs ~20-40 s for the real model), so the
tracemalloc probe finishes in seconds.

Run:   python mini_leak_repro.py
Tune:  N, T, MAX_STEP/ATOL/RTOL, CELLS at the bottom.
Read:  rss climbs while `python` stays flat  -> allocator fragmentation
       (same signature as the real model; fix = worker recycling / malloc_trim).
       neither climbs                          -> the leak needs the real
       model's allocation scale (still informative -> fragmentation).
       `python` climbs                          -> a real Python leak; the
       compare_to lines name the growing site.

The model + run_point are module-level, so you can also import run_point into a
pool test (run_parallel_tasks / run_xso_parscan) for the serial-vs-parallel A/B.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import xso


@xso.component
class Vector:
    """Holds the N-class state vector and its size-class index."""
    value = xso.variable(dims='sp', description='state per class')
    sp = xso.index(dims='sp', description='class index')


@xso.component
class Logistic:
    """Competition dynamics; the flux allocates an N x N temp per RHS call."""
    var = xso.variable(foreign=True, dims='sp', flux='grow', negative=False)
    r = xso.parameter(dims='sp', description='intrinsic growth rates')
    comp = xso.parameter(description='competition coefficient')

    @xso.flux(dims='sp')
    def grow(self, var, r, comp):
        outer = var[:, None] * var[None, :]       # N x N temporary, every call
        return var * r - comp * outer.sum(axis=1)


model = xso.create({'Pop': Vector, 'Growth': Logistic})


def run_point(N=80, T=14601.0, dt=1.0, max_step=1.0, atol=1e-9, rtol=1e-6):
    """One cell: build setup, solve, reduce to a small dict (mimics run_one)."""
    rng = np.random.default_rng(0)                # fixed seed -> deterministic
    r = 0.4 + 0.05 * rng.standard_normal(N)
    setup = xso.setup(
        solver='solve_ivp', model=model, time=np.arange(0.0, T, dt),
        input_vars={
            'Pop':    {'value_label': 'P', 'value_init': np.full(N, 0.1),
                       'sp_index': np.arange(N)},
            'Growth': {'var': 'P', 'r': r, 'comp': 0.02},
        },
        output_vars={'Pop__value'},               # SLIM, like your real scan
        solver_kwargs={'method': 'RK45', 'atol': atol, 'rtol': rtol,
                       'max_step': max_step},
    )
    with model:
        out = setup.xsimlab.run()
    return {'final_sum': float(out['Pop__value'].isel(time=-1).sum().values)}


if __name__ == "__main__":
    import gc
    import tracemalloc
    import psutil

    N, T, CELLS = 80, 14601.0, 20
    print(f"mini leak repro: N={N}, T={T}, {CELLS} cells, serial single process")

    tracemalloc.start()
    proc, prev = psutil.Process(os.getpid()), None
    for i in range(CELLS):
        result = run_point(N=N, T=T)
        del result
        gc.collect()

        py = tracemalloc.get_traced_memory()[0] / 1e6
        rss = proc.memory_info().rss / 1e6
        print(f"cell {i:2d}: python={py:7.1f} MB | rss={rss:7.1f} MB", flush=True)

        snap = tracemalloc.take_snapshot()
        if prev:
            for s in snap.compare_to(prev, 'lineno')[:4]:
                print("   ", s)
        prev = snap
