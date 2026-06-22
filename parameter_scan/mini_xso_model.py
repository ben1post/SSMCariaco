"""
mini_xso_model.py -- a clean, minimal, NON-chaotic XSO model for testing the
parallel-scan mechanics in isolation from the real Cariaco model.

Model: Monod NP chemostat (handoff section 14.1). Two state variables, globally
stable -> converges to a steady state, no stiffness, no blow-ups, fully
deterministic for a given parameter set. A single run solves in milliseconds;
per-task cost is dialed UP deliberately via MAX_STEP so the speedup test has
something to amortise.

Why a separate module (not a notebook cell): macOS multiprocessing uses spawn,
so every worker RE-IMPORTS the worker's module. Keeping the model + worker here,
lightweight and importable (no matplotlib, no Cariaco stack, no __main__
coupling), is exactly what stops the Pool from hanging.
"""
import os

# Pin BLAS/OMP before numpy import -- runs here in the parent AND in every spawned
# worker (which re-imports this module before its own numpy import).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import xso


# ----------------------------- components ----------------------------------
@xso.component
class StateVariable:
    value = xso.variable(description="concentration", attrs={"units": "mmol N m-3"})


@xso.component
class ConstantExternalNutrient:
    forcing = xso.forcing(setup_func="forcing_setup", description="external nutrient")
    value = xso.parameter(description="constant value")

    def forcing_setup(self, value):
        @np.vectorize
        def f(t):
            return value
        return f


@xso.component
class LinearInflow:
    sink = xso.variable(foreign=True, flux="input", negative=False)
    source = xso.forcing(foreign=True)
    rate = xso.parameter(description="inflow rate")

    @xso.flux
    def input(self, sink, source, rate):
        return source * rate


@xso.component
class LinearOutflow_ListInput:
    var_list = xso.variable(dims="flow_list", list_input=True, foreign=True,
                            flux="decay", negative=True)
    rate = xso.parameter(description="outflow rate")

    @xso.flux(dims="flow_list")
    def decay(self, var_list, rate):
        return var_list * rate


@xso.component
class MonodGrowth:
    resource = xso.variable(foreign=True, flux="uptake", negative=True)
    consumer = xso.variable(foreign=True, flux="uptake", negative=False)
    halfsat = xso.parameter(description="half-saturation")
    mu_max = xso.parameter(description="maximum growth rate")

    @xso.flux
    def uptake(self, resource, consumer, halfsat, mu_max):
        return mu_max * resource / (resource + halfsat) * consumer


# Built ONCE at import -> reused by every task a worker handles (the amortisation
# that makes the pool worthwhile), exactly like model_baseline_seasonal.
model = xso.create({
    "Nutrient": StateVariable,
    "Phytoplankton": StateVariable,
    "N0": ConstantExternalNutrient,
    "Inflow": LinearInflow,
    "Outflow": LinearOutflow_ListInput,
    "Growth": MonodGrowth,
})


# ------------------------------- worker ------------------------------------
def run_point(mu_max, t_max=400.0, dt=0.1, max_step=0.02):
    """One independent run at a given mu_max -> small picklable summary dict.

    Mirrors run_one: builds its OWN fresh setup, solves, reduces to a tiny dict.
    `max_step` is the cost dial -- smaller forces more RHS evaluations so each
    task takes measurable wall time (the chemostat itself is trivial). The result
    is deterministic; mu_max only shifts the steady state, never destabilises it.
    """
    setup = xso.setup(
        solver="solve_ivp", model=model,
        time=np.arange(0.0, t_max, dt),
        input_vars={
            "Nutrient":      {"value_label": "N", "value_init": 1.0},
            "Phytoplankton": {"value_label": "P", "value_init": 0.1},
            "N0":            {"forcing_label": "N0", "value": 1.0},
            "Inflow":        {"source": "N0", "sink": "N", "rate": 0.1},
            "Outflow":       {"var_list": ["N", "P"], "rate": 0.1},
            "Growth":        {"resource": "N", "consumer": "P",
                              "halfsat": 0.7, "mu_max": float(mu_max)},
        },
        solver_kwargs={"max_step": max_step},
    )
    with model:
        out = setup.xsimlab.run()
    return {
        "mu_max": float(mu_max),
        "N_final": float(out["Nutrient__value"].isel(time=-1).values),
        "P_final": float(out["Phytoplankton__value"].isel(time=-1).values),
    }
