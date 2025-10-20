import xso
import numpy as np
from multiprocessing import Pool

# ------------------------------------------------------------
# 1️⃣ Define components (importable or defined at top level)
# ------------------------------------------------------------

@xso.component
class Variable:
    var = xso.variable(description='basic state variable', attrs={'units':'µM'})

@xso.component
class LinearGrowth:
    var_ext = xso.variable(foreign=True, flux='growth', description='external state variable')
    rate = xso.parameter(description='linear growth rate', attrs={'units':'$d^{-1}$'})

    @xso.flux
    def growth(self, var_ext, rate):
        return var_ext * rate

# ------------------------------------------------------------
# 2️⃣ Define the worker function
# ------------------------------------------------------------

def run_model_parallel(components, time, input_vars, param_overrides=None):
    """
    Build and run an xso model inside a worker.
    
    Parameters
    ----------
    components : dict
        Mapping of component names to xso component classes.
    time : array-like
        Time array for simulation.
    input_vars : dict
        Dictionary of input variables for xso.setup().
    param_overrides : dict, optional
        Optional dict of overrides (e.g. {'Growth': {'rate': 0.5}}).
    """
    import xso
    import numpy as np

    # Create model
    model = xso.create(components, time_unit='d')

    # Apply parameter overrides
    input_vars_local = {k: v.copy() for k, v in input_vars.items()}
    if param_overrides:
        for comp, overrides in param_overrides.items():
            input_vars_local.setdefault(comp, {})
            input_vars_local[comp].update(overrides)

    # Setup model
    ds = xso.setup(
        solver='solve_ivp',
        model=model,
        time=np.asarray(time),
        input_vars=input_vars_local,
    )

    # Run model
    with model:
        out = ds.xsimlab.run()

    # Round time to avoid float precision issues
    out['time'] = out.time.round(9)
    return out