"""
Parameter Scan Utilities
========================
Cost functions, model→target aggregation, 2D cost grid computation,
and best-fit extraction for CARIACO size-spectrum parameter scans.
"""

import numpy as np


# =============================================================================
# SIZE-BIN HELPERS
# =============================================================================
def get_log_bin_edges(centers):
    """Compute N+1 log-spaced bin edges for N log-spaced bin centers."""
    q = centers[1] / centers[0]
    half_step = np.sqrt(q)
    edges = np.zeros(len(centers) + 1)
    edges[0] = centers[0] / half_step
    edges[1:] = centers * half_step
    return edges


def get_fraction_in_range(lower, upper, target_min, target_max):
    """Log-space fractional overlap of bin [lower, upper] with [target_min, target_max]."""
    overlap_min = max(lower, target_min)
    overlap_max = min(upper, target_max)
    if overlap_min >= overlap_max:
        return 0.0
    return ((np.log10(overlap_max) - np.log10(overlap_min))
            / (np.log10(upper) - np.log10(lower)))


# =============================================================================
# MODEL → TARGET AGGREGATION
# =============================================================================
def aggregate_model_to_targets(model_state, phyto_esd, zoo_esd,
                               bin_definitions, d_e=None):
    """
    Aggregate model output onto observation targets defined by bin_definitions.

    Parameters
    ----------
    model_state : dict
        Model state at the evaluation time. Expected keys by target type:
            'phyto'    -> 1D array over phyto size classes
            'zoo'      -> 1D array over zoo size classes
            'nutrient' -> scalar
            'detritus' -> scalar (PON)
            'export'   -> scalar (volumetric sinking flux, mmol N m-3 d-1;
                                  multiplied by d_e to give areal flux
                                  matching trap observations)
    phyto_esd : array-like
        Phyto size-class centers (µm ESD).
    zoo_esd : array-like
        Zoo size-class centers (µm ESD).
    bin_definitions : list of dict
        Target bin definitions from cariaco_obs.TARGET_BIN_DEFINITIONS.
    d_e : float, optional
        Euphotic-zone box depth [m]. Required if any 'export' targets
        are present (used to convert volumetric sinking flux to areal).

    Returns
    -------
    model_vec : np.ndarray, shape (n_targets,)
        Model values in the same order as bin_definitions.
    """
    p_edges = get_log_bin_edges(np.asarray(phyto_esd))
    z_edges = get_log_bin_edges(np.asarray(zoo_esd))

    model_vec = np.zeros(len(bin_definitions))

    for k, b in enumerate(bin_definitions):
        t = b['type']

        if t == 'phyto':
            total = 0.0
            for i in range(len(phyto_esd)):
                total += model_state['phyto'][i] * get_fraction_in_range(
                    p_edges[i], p_edges[i + 1], b['size_min'], b['size_max'])
            model_vec[k] = total

        elif t == 'zoo':
            total = 0.0
            for i in range(len(zoo_esd)):
                total += model_state['zoo'][i] * get_fraction_in_range(
                    z_edges[i], z_edges[i + 1], b['size_min'], b['size_max'])
            model_vec[k] = total

        elif t == 'nutrient':
            model_vec[k] = float(model_state['nutrient'])

        elif t == 'detritus':
            model_vec[k] = float(model_state['detritus'])

        elif t == 'export':
            if d_e is None:
                raise ValueError(
                    "d_e must be provided to aggregate 'export' targets.")
            # volumetric flux [mmol N m-3 d-1] * box depth [m]
            #   -> areal flux [mmol N m-2 d-1]  (matches trap obs)
            model_vec[k] = float(model_state['export']) * d_e

        else:
            raise ValueError(
                f"Unknown target type '{t}' in bin_definitions. Supported: "
                f"'phyto', 'zoo', 'nutrient', 'detritus', 'export'.")

    return model_vec


# =============================================================================
# COST FUNCTION
# =============================================================================
def compute_cost_nrmsre(model_vec, obs_vec):
    """
    Normalized Root Mean Square Relative Error.

        cost = sqrt( (1/N) * Σ ((model_i - obs_i) / obs_i)^2 )

    Dimensionless — all targets weighted equally regardless of magnitude.
    Cost ~0.3 means "on average ~30% relative error across targets."
    """
    rel_errors = (model_vec - obs_vec) / obs_vec
    return np.sqrt(np.mean(rel_errors ** 2))

# =============================================================================
# 2D COST GRID
# =============================================================================
def compute_cost_grid(scan_results, phyto_esd, zoo_esd, obs_vec,
                      bin_definitions, avg_window, dim1_name, dim2_name,
                      reject_negative=True):
    """
    Post-process an xso.parscans 2D scan into a cost grid.

    Parameters
    ----------
    scan_results : xarray.Dataset
        Output from run_xso_parscan. Must contain the model variables
        needed by the target types present in bin_definitions:
            'phyto'    -> 'Phytoplankton__biomass'
            'zoo'      -> 'Zooplankton__biomass'
            'nutrient' -> 'Nutrient__value'
            'detritus' -> 'Detritus__value'
            'export'   -> 'DetritusSink__sinking_value'
        All must have a 'time' dimension and the two scan dimensions.
        For 'export' targets, scan_results must also carry 'Inflow__de'
        as a coordinate (attached automatically when Inflow__de is
        passed via fixed_overrides to run_xso_parscan).
    phyto_esd, zoo_esd : arrays
        Size class centers (µm ESD).
    obs_vec : np.ndarray, shape (n_targets,)
        Observation target vector.
    bin_definitions : list of dict
        Target bin definitions (must match obs_vec ordering).
    avg_window : int
        Number of final timesteps to average over (steady-state window).
    dim1_name, dim2_name : str
        Names of the two scan dimensions in scan_results.
    reject_negative : bool
        If True, runs with any negative value in the averaged state
        are flagged as failed (cost = NaN). Ecological models cannot
        produce negative values — negatives indicate numerical breakdown.

    Returns
    -------
    cost_grid : np.ndarray, shape (n1, n2)
        Cost for each (param1, param2) combination. NaN for failed runs.
    model_grid : np.ndarray, shape (n1, n2, n_targets)
        Model target vector for each combination. NaN rows for failed runs.
    """
    tail = slice(-avg_window, None)
    types_needed = set(b['type'] for b in bin_definitions)

    # Map target type -> (scan_results variable name, model_state key)
    var_map = {
        'phyto':    ('Phytoplankton__biomass',      'phyto'),
        'zoo':      ('Zooplankton__biomass',        'zoo'),
        'nutrient': ('Nutrient__value',             'nutrient'),
        'detritus': ('Detritus__value',             'detritus'),
        'export':   ('DetritusSink__sinking_value', 'export'),
    }

    averaged = {}
    for t in types_needed:
        if t not in var_map:
            raise ValueError(
                f"Unknown target type '{t}' in bin_definitions. "
                f"Supported: {list(var_map)}."
            )
        ds_name, state_key = var_map[t]
        averaged[state_key] = scan_results[ds_name].isel(time=tail).mean('time')

    # Pull d_e from the scan coords (attached automatically when Inflow__de
    # is passed via fixed_overrides). Only needed if 'export' targets exist.
    d_e = None
    if 'export' in types_needed:
        if 'Inflow__de' not in scan_results.coords:
            raise ValueError(
                "Bin definitions contain 'export' targets, but scan_results "
                "has no 'Inflow__de' coordinate. Pass Inflow__de via "
                "fixed_overrides when running the parscan so d_e is recorded "
                "alongside the scan."
            )
        d_e = float(scan_results['Inflow__de'].values)

    n1 = len(scan_results[dim1_name])
    n2 = len(scan_results[dim2_name])
    n_targets = len(bin_definitions)

    cost_grid = np.full((n1, n2), np.nan)
    model_grid = np.full((n1, n2, n_targets), np.nan)

    for i in range(n1):
        for j in range(n2):
            model_state = {}
            bad = False
            for key, arr in averaged.items():
                val = arr.isel({dim1_name: i, dim2_name: j}).values
                if np.any(np.isnan(val)):
                    bad = True
                    break
                if reject_negative and np.any(val < 0):
                    bad = True
                    break
                model_state[key] = val
            if bad:
                continue

            model_vec = aggregate_model_to_targets(
                model_state, phyto_esd, zoo_esd, bin_definitions, d_e=d_e,
            )
            model_grid[i, j, :] = model_vec
            cost_grid[i, j] = compute_cost_nrmsre(model_vec, obs_vec)

    return cost_grid, model_grid

# =============================================================================
# BEST-FIT EXTRACTION
# =============================================================================
def find_best_fit(cost_grid, model_grid, scan_results, dim1_name, dim2_name):
    """
    Find the (param1, param2) combination with minimum cost.

    Returns
    -------
    best : dict
        {
          'cost':      float,
          'idx':       (i, j),
          'val1':      value of dim1 at best fit,
          'val2':      value of dim2 at best fit,
          'model_vec': model target vector at best fit,
        }
    """
    if np.all(np.isnan(cost_grid)):
        raise ValueError("All runs failed — cost_grid is entirely NaN.")

    idx = np.unravel_index(np.nanargmin(cost_grid), cost_grid.shape)
    return {
        'cost':      cost_grid[idx],
        'idx':       idx,
        'val1':      float(scan_results[dim1_name].values[idx[0]]),
        'val2':      float(scan_results[dim2_name].values[idx[1]]),
        'model_vec': model_grid[idx[0], idx[1], :],
    }


# =============================================================================
# DEFAULT-PARAMETER LOOKUP (for plot markers)
# =============================================================================
def get_default_from_setup(model_setup, param_name):
    """
    Read a scalar default parameter value from an XSO model_setup object.

    Parameters
    ----------
    model_setup : xso setup object
        The object created by xso.setup(...).
    param_name : str
        Fully-qualified XSO parameter name, e.g. 'Grazing__KsZ'.

    Returns
    -------
    value : float
        The scalar default value of that parameter.
    """
    return float(model_setup[param_name].values)