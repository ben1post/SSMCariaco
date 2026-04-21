"""
Parameter Scan Utilities
========================
Cost functions, model→target aggregation, 2D cost grid computation,
and best-fit extraction for CARIACO size-spectrum parameter scans.
"""

import numpy as np


# =============================================================================
# TARGET TYPE REGISTRY
# =============================================================================
# Single source of truth for how each target type is extracted from scan
# results. Adding a new target type requires:
#   1. An entry in TARGET_EXTRACTORS below (how to extract it)
#   2. An 'aggregate' branch in aggregate_model_to_targets (how to collapse
#      it to a scalar against obs)
#   3. (Plotting only) entries in parscan_plots.TYPE_COLOR_PALETTES
#      and TYPE_UNITS
#
# Per-type entry schema:
#   'ivp'    : str — name of the XSO output variable in an IVP scan
#              (will be time-averaged over the tail window in the caller).
#   'steady' : either
#                (a) str — name of a state variable present in the
#                    stability scan output (evaluated at time=-1), or
#                (b) callable (stability_results, steady_ds) -> DataArray
#                    — analytical reconstruction from steady-state state
#                    + parameters, returning a DataArray broadcastable
#                    against the scan dims (and, where applicable, the
#                    size-class dim).
TARGET_EXTRACTORS = {
    'phyto':    {'ivp': 'Phytoplankton__biomass',
                 'steady': 'Phytoplankton__biomass'},
    'zoo':      {'ivp': 'Zooplankton__biomass',
                 'steady': 'Zooplankton__biomass'},
    'nutrient': {'ivp': 'Nutrient__value',
                 'steady': 'Nutrient__value'},
    'detritus': {'ivp': 'Detritus__value',
                 'steady': 'Detritus__value'},
    'export':   {'ivp': 'DetritusSink__sinking_value',
                 'steady': lambda ds, steady: (
                     ds['DetritusSink__sinking_rate']
                     * steady['Detritus__value'])},
    'pp':       {'ivp': 'Growth__uptake_value',
                 'steady': lambda ds, steady: (
                     ds['Growth__mu_max']
                     * steady['Nutrient__value']
                     / (steady['Nutrient__value']
                        + ds['Growth__halfsat'])
                     * steady['Phytoplankton__biomass'])},
}

# Types whose aggregation needs d_e (euphotic-zone box depth) in model_state.
TYPES_REQUIRING_DE = {'export'}


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
                               bin_definitions):
    """
    Aggregate model output onto observation targets defined by bin_definitions.

    Parameters
    ----------
    model_state : dict
        Model state at the evaluation time. Expected keys by target type:
            'phyto'    -> 1D array over phyto size classes
            'zoo'      -> 1D array over zoo size classes
            'nutrient' -> scalar
            'detritus' -> scalar
            'pp'       -> 1D array over phyto size classes (uptake flux)
            'export'   -> scalar (volumetric sinking flux, mmol N m-3 d-1)
            'd_e'      -> scalar euphotic-zone box depth [m]; required if
                          any type in TYPES_REQUIRING_DE is present.
    phyto_esd, zoo_esd : array-like
        Size-class centers (µm ESD).
    bin_definitions : list of dict
        Target bin definitions.

    Returns
    -------
    model_vec : np.ndarray, shape (n_targets,)
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

        elif t == 'pp':
            # Total phyto uptake summed across size classes [mmol N m-3 d-1]
            model_vec[k] = float(np.sum(model_state['pp']))

        elif t == 'export':
            if 'd_e' not in model_state:
                raise ValueError(
                    "model_state must contain 'd_e' to aggregate "
                    "'export' targets.")
            # volumetric flux [mmol N m-3 d-1] * box depth [m]
            #   -> areal flux [mmol N m-2 d-1]  (matches trap obs)
            model_vec[k] = (float(model_state['export'])
                            * float(model_state['d_e']))

        else:
            raise ValueError(
                f"Unknown target type '{t}' in bin_definitions. "
                f"Supported: {sorted(TARGET_EXTRACTORS)}."
            )

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
# SHARED CELL-ITERATION HELPER
# =============================================================================
def _iterate_cost_grid(state_das, n1, n2, dim1_name, dim2_name,
                       bin_definitions, obs_vec, phyto_esd, zoo_esd,
                       d_e_scalar=None, d_e_da=None,
                       stable_mask=None,
                       neg_tolerance=0.0,
                       clip_small_negatives=False):
    """
    Iterate over all (i, j) scan cells and build cost_grid / model_grid.

    Shared inner loop used by both compute_cost_grid (IVP) and
    compute_cost_grid_steady_state. For each cell:
      - optionally skip if `stable_mask` is False there
      - pull each DataArray in `state_das` at (i, j); bad-check for
        NaN or values below -neg_tolerance; optionally clip small negatives
        in [-neg_tolerance, 0) to 0
      - set model_state['d_e'] from d_e_scalar (fixed) or d_e_da (varying)
      - aggregate to target vector and compute NRMSRE cost

    Parameters
    ----------
    state_das : dict[str, xarray.DataArray]
        Keys are model_state keys for aggregate_model_to_targets
        ('phyto', 'zoo', 'nutrient', 'detritus', 'pp', 'export'). Each
        DataArray has scan dims (dim1_name, dim2_name) and possibly an
        extra size-class dim ('phyto' or 'zoo').
    n1, n2 : int
        Sizes of the two scan dimensions.
    dim1_name, dim2_name : str
    bin_definitions, obs_vec, phyto_esd, zoo_esd :
        Passed through to aggregate_model_to_targets and cost.
    d_e_scalar : float or None
        Fixed d_e for all cells (preferred if forcing is uniform).
    d_e_da : xarray.DataArray or None
        Scan-dim-varying d_e; used only if d_e_scalar is None.
    stable_mask : xarray.DataArray or None
        Boolean mask over scan dims; cells where False are skipped.
    neg_tolerance : float
        Values below -neg_tolerance fail the cell (cost = NaN).
        Default 0.0 = any negative fails.
    clip_small_negatives : bool
        If True, values in [-neg_tolerance, 0) are clipped to 0 before
        aggregation. Use with a nonzero neg_tolerance for steady-state
        scans where fsolve produces floating-point noise around zero.
    """
    n_targets = len(bin_definitions)
    cost_grid = np.full((n1, n2), np.nan)
    model_grid = np.full((n1, n2, n_targets), np.nan)

    for i in range(n1):
        for j in range(n2):
            if stable_mask is not None:
                if not bool(stable_mask.isel(
                        {dim1_name: i, dim2_name: j}).values):
                    continue

            model_state = {}
            bad = False
            for key, arr in state_das.items():
                val = arr.isel({dim1_name: i, dim2_name: j}).values
                if np.any(np.isnan(val)):
                    bad = True
                    break
                if np.any(val < -neg_tolerance):
                    bad = True
                    break
                if clip_small_negatives:
                    val = np.clip(val, 0.0, None)
                model_state[key] = val
            if bad:
                continue

            if d_e_scalar is not None:
                model_state['d_e'] = d_e_scalar
            elif d_e_da is not None:
                model_state['d_e'] = float(
                    d_e_da.isel({dim1_name: i, dim2_name: j}).values
                )

            model_vec = aggregate_model_to_targets(
                model_state, phyto_esd, zoo_esd, bin_definitions,
            )
            model_grid[i, j, :] = model_vec
            cost_grid[i, j] = compute_cost_nrmsre(model_vec, obs_vec)

    return cost_grid, model_grid


# =============================================================================
# 2D COST GRID — IVP SCAN
# =============================================================================
def compute_cost_grid(scan_results, phyto_esd, zoo_esd, obs_vec,
                      bin_definitions, avg_window, dim1_name, dim2_name,
                      reject_negative=True):
    """
    Post-process an xso.parscans 2D IVP scan into a cost grid.

    For each cell, each target type's XSO output variable (see
    TARGET_EXTRACTORS[t]['ivp']) is averaged over the last `avg_window`
    timesteps, then aggregated to the target vector and scored against
    obs_vec.

    Parameters
    ----------
    scan_results : xarray.Dataset
        Must contain each TARGET_EXTRACTORS[t]['ivp'] variable with a
        'time' dim and both scan dims. If any 'export' target is present,
        must also contain 'Inflow__de'.
    avg_window : int
        Number of final timesteps to average.
    reject_negative : bool
        If True, cells with any negative averaged value are flagged
        as failed (cost = NaN).

    Returns
    -------
    cost_grid : np.ndarray, shape (n1, n2)
    model_grid : np.ndarray, shape (n1, n2, n_targets)
    """
    tail = slice(-avg_window, None)
    types_needed = set(b['type'] for b in bin_definitions)

    # Average each required output variable over the tail window.
    state_das = {}
    for t in types_needed:
        if t not in TARGET_EXTRACTORS:
            raise ValueError(
                f"Unknown target type '{t}' in bin_definitions. "
                f"Supported: {sorted(TARGET_EXTRACTORS)}."
            )
        var_name = TARGET_EXTRACTORS[t]['ivp']
        if var_name not in scan_results.variables:
            raise ValueError(
                f"Target type '{t}' requires '{var_name}' in scan_results "
                f"but it is not present. Add it to output_vars in the "
                f"model setup."
            )
        state_das[t] = (scan_results[var_name]
                        .isel(time=tail).mean('time'))

    # d_e: scalar (fixed via overrides) or scan-dim-varying.
    d_e_scalar = None
    d_e_da = None
    if TYPES_REQUIRING_DE & types_needed:
        if 'Inflow__de' not in scan_results.variables:
            raise ValueError(
                "Bin definitions contain target(s) in TYPES_REQUIRING_DE, "
                "but scan_results does not contain 'Inflow__de'. Make sure "
                "the Inflow component is part of your model setup."
            )
        d_e_da = scan_results['Inflow__de']
        if d_e_da.ndim == 0:
            d_e_scalar = float(d_e_da.values)
            d_e_da = None

    n1 = len(scan_results[dim1_name])
    n2 = len(scan_results[dim2_name])

    # reject_negative=True  -> neg_tolerance=0.0  (any negative fails)
    # reject_negative=False -> neg_tolerance=+inf (negatives accepted)
    neg_tol = 0.0 if reject_negative else np.inf

    return _iterate_cost_grid(
        state_das, n1, n2, dim1_name, dim2_name,
        bin_definitions, obs_vec, phyto_esd, zoo_esd,
        d_e_scalar=d_e_scalar, d_e_da=d_e_da,
        stable_mask=None,
        neg_tolerance=neg_tol,
        clip_small_negatives=False,
    )


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
    """
    return float(model_setup[param_name].values)


# =============================================================================
# STEADY-STATE SEED EXTRACTION
# =============================================================================
def extract_steady_state_seed(scan_results, avg_window):
    """
    Build an initial-value seed dataset + iv_mapping from an IVP scan,
    by averaging each state variable over the last `avg_window` timesteps.

    Intended to feed into run_xso_stabilityscan(..., initial_values_ds=seed_ds,
    iv_mapping=iv_map).

    State variables are hardcoded for the CARIACO NPZD setup:
        Nutrient__value, Phytoplankton__biomass,
        Zooplankton__biomass, Detritus__value
    """
    state_vars = [
        'Nutrient__value',
        'Phytoplankton__biomass',
        'Zooplankton__biomass',
        'Detritus__value',
    ]

    seed_ds = (scan_results[state_vars]
               .isel(time=slice(-avg_window, None))
               .mean('time'))

    iv_mapping = {v: v + '_init' for v in state_vars}

    return seed_ds, iv_mapping


# =============================================================================
# 2D COST GRID — STEADY STATE (from stability scan)
# =============================================================================
def compute_cost_grid_steady_state(stability_results, phyto_esd, zoo_esd,
                                   obs_vec, bin_definitions,
                                   dim1_name, dim2_name,
                                   neg_tolerance=1e-6,
                                   require_stable=True):
    """
    Post-process an xso.parscans stability scan into a cost grid.

    For each cell, each target type's extractor (see
    TARGET_EXTRACTORS[t]['steady']) is evaluated at the fsolve steady
    state (time=-1). Extractors are either a variable name (for state
    variables directly present in stability_results) or a callable
    (stability_results, steady_ds) -> DataArray for analytical
    reconstruction of fluxes not stored in the scan output.

    A cell becomes NaN in the cost grid if:
      - require_stable=True and stability != 'stable'
      - any extracted value is < -neg_tolerance (true negative, not noise)
      - any extracted value is NaN (fsolve failed)

    Small negative values in [-neg_tolerance, 0) are clipped to 0.

    Returns
    -------
    cost_grid : np.ndarray, shape (n1, n2)
    model_grid : np.ndarray, shape (n1, n2, n_targets)
    """
    steady = stability_results.isel(time=-1)
    types_needed = set(b['type'] for b in bin_definitions)

    state_das = {}
    for t in types_needed:
        if t not in TARGET_EXTRACTORS:
            raise ValueError(
                f"Unknown target type '{t}' in bin_definitions. "
                f"Supported: {sorted(TARGET_EXTRACTORS)}."
            )
        extractor = TARGET_EXTRACTORS[t]['steady']
        if isinstance(extractor, str):
            if extractor not in stability_results.variables:
                raise ValueError(
                    f"Target type '{t}' requires '{extractor}' in "
                    f"stability_results but it is not present."
                )
            state_das[t] = steady[extractor]
        elif callable(extractor):
            try:
                state_das[t] = extractor(stability_results, steady)
            except KeyError as e:
                raise ValueError(
                    f"Target type '{t}' requires analytical reconstruction "
                    f"from stability_results, but a required variable is "
                    f"missing: {e}"
                ) from e
        else:
            raise TypeError(
                f"TARGET_EXTRACTORS['{t}']['steady'] must be a str or "
                f"callable, got {type(extractor).__name__}."
            )

    # d_e: always scalar in steady-state scans (forcing is fixed).
    d_e_scalar = None
    if TYPES_REQUIRING_DE & types_needed:
        if 'Inflow__de' not in stability_results.variables:
            raise ValueError(
                "Bin definitions contain target(s) in TYPES_REQUIRING_DE, "
                "but stability_results does not contain 'Inflow__de'."
            )
        d_e_scalar = float(stability_results['Inflow__de'].values)

    stable_mask = (stability_results['stability'] == 'stable'
                   if require_stable else None)

    n1 = len(stability_results[dim1_name])
    n2 = len(stability_results[dim2_name])

    return _iterate_cost_grid(
        state_das, n1, n2, dim1_name, dim2_name,
        bin_definitions, obs_vec, phyto_esd, zoo_esd,
        d_e_scalar=d_e_scalar, d_e_da=None,
        stable_mask=stable_mask,
        neg_tolerance=neg_tolerance,
        clip_small_negatives=True,
    )