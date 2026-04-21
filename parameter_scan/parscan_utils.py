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
# Single source of truth for which XSO output variable each target type
# reads from. Adding a new target type requires:
#   1. An entry here (how to extract it from scan output)
#   2. An 'aggregate' branch in aggregate_model_to_targets (how to collapse
#      the extracted value to a scalar against obs)
#   3. (Plotting only) entries in parscan_plots.TYPE_COLOR_PALETTES
#      and TYPE_UNITS
#
# The same variable name is used for both IVP and steady-state scans.
# The caller handles temporal aggregation (time-averaging for IVP,
# time=-1 selection for stability).
TARGET_EXTRACTORS = {
    'phyto':    'Phytoplankton__biomass',
    'zoo':      'Zooplankton__biomass',
    'nutrient': 'Nutrient__value',
    'detritus': 'Detritus__value',
    'export':   'DetritusSink__sinking_value',
    'pp':       'Growth__uptake_value',
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


def _build_state_das(results_ds, bin_definitions, time_collapser):
    """Extract target-type DataArrays from scan output, collapsing time.

    Walks over the unique target types in `bin_definitions`, looks each one
    up in TARGET_EXTRACTORS to find its XSO output variable name, and
    applies `time_collapser` to reduce the time dimension to a single value
    per scan cell.

    The returned dict uses the target type as the key (not the XSO variable
    name), matching the `model_state` keys expected by
    `aggregate_model_to_targets`.

    Parameters
    ----------
    results_ds : xarray.Dataset
        Scan output — either from `run_xso_parscan` (IVP) or
        `run_xso_stabilityscan` (steady state). Must contain each XSO
        variable named in TARGET_EXTRACTORS for the types used.
    bin_definitions : list of dict
        Target definitions; only the 'type' field is read here.
    time_collapser : callable
        DataArray -> DataArray reducing the 'time' dim. Typical choices:
          IVP:      lambda da: da.isel(time=slice(-avg_window, None)).mean('time')
          Steady:   lambda da: da.isel(time=-1)

    Returns
    -------
    state_das : dict[str, xarray.DataArray]
        Keys are target types ('phyto', 'zoo', ...); values are DataArrays
        with scan dims (and any extra model dim like 'phyto' or 'zoo'),
        time already collapsed.

    Raises
    ------
    ValueError
        If a target type in `bin_definitions` is not in TARGET_EXTRACTORS,
        or if its XSO output variable is missing from `results_ds`
        (typically because it was not added to `output_vars` in the setup).
    """
    state_das = {}
    for t in set(b['type'] for b in bin_definitions):
        if t not in TARGET_EXTRACTORS:
            raise ValueError(
                f"Unknown target type '{t}' in bin_definitions. "
                f"Supported: {sorted(TARGET_EXTRACTORS)}."
            )
        var_name = TARGET_EXTRACTORS[t]
        if var_name not in results_ds.variables:
            raise ValueError(
                f"Target type '{t}' requires '{var_name}' in results_ds "
                f"but it is not present."
            )
        state_das[t] = time_collapser(results_ds[var_name])
    return state_das


def _resolve_de(results_ds, types_needed):
    """Resolve the euphotic-zone box depth (d_e) for aggregation.

    Some target types (those in TYPES_REQUIRING_DE, e.g. 'export') need
    d_e to convert volumetric model fluxes to areal observation units.
    d_e may have been supplied as a fixed value (via `fixed_overrides` at
    scan time) or may vary across the scan grid. This helper normalises
    both cases into a pair (scalar, DataArray) where exactly one is set.

    If no type in `types_needed` requires d_e, both returns are None.

    Parameters
    ----------
    results_ds : xarray.Dataset
        Must contain 'Inflow__de' if any type in `types_needed` requires it.
    types_needed : set of str
        Target types present in the current bin_definitions.

    Returns
    -------
    d_e_scalar : float or None
        Set (and `d_e_da` is None) if d_e is a scalar — stored as a 0-D
        DataArray in `results_ds` because it was fixed across the scan.
    d_e_da : xarray.DataArray or None
        Set (and `d_e_scalar` is None) if d_e varies across the scan
        dims. Caller indexes it per cell.

    Raises
    ------
    ValueError
        If `types_needed` contains a type in TYPES_REQUIRING_DE but
        'Inflow__de' is not in `results_ds`.
    """
    if not (TYPES_REQUIRING_DE & types_needed):
        return None, None
    if 'Inflow__de' not in results_ds.variables:
        raise ValueError(
            "Bin definitions contain target(s) in TYPES_REQUIRING_DE, but "
            "'Inflow__de' is not in results_ds."
        )
    d_e_da = results_ds['Inflow__de']
    if d_e_da.ndim == 0:
        return float(d_e_da.values), None
    return None, d_e_da

# =============================================================================
# 2D COST GRID — IVP SCAN
# =============================================================================
def compute_cost_grid(scan_results, phyto_esd, zoo_esd, obs_vec,
                      bin_definitions, avg_window, dim1_name, dim2_name,
                      reject_negative=True):
    """Post-process an xso.parscans 2D IVP scan into a cost grid.

    For each cell in the (dim1_name × dim2_name) scan grid, each target
    type's XSO output variable (see TARGET_EXTRACTORS) is averaged over
    the last `avg_window` timesteps, then aggregated into the target
    vector via `aggregate_model_to_targets` and scored against `obs_vec`
    using NRMSRE.

    Cells with any negative averaged value are flagged as failed
    (cost = NaN) when `reject_negative=True`. This is the appropriate
    default for IVP output — true-zero state variables can't go negative
    in a correctly specified NPZD system, so negatives indicate solver
    instability rather than floating-point noise.

    Parameters
    ----------
    scan_results : xarray.Dataset
        Output from `run_xso_parscan`. Must contain each XSO variable
        named in TARGET_EXTRACTORS for the types used in `bin_definitions`,
        each with a 'time' dim and both scan dims. If any 'export' target
        is present, must also contain 'Inflow__de'.
    phyto_esd, zoo_esd : array-like
        Size-class centers (µm ESD).
    obs_vec : np.ndarray, shape (n_targets,)
        Observation target vector.
    bin_definitions : list of dict
        Target bin definitions (must match obs_vec ordering).
    avg_window : int
        Number of final timesteps to average over (steady-state window).
    dim1_name, dim2_name : str
        Names of the two scan dimensions in `scan_results`.
    reject_negative : bool
        If True (default), cells with any negative averaged value are
        flagged as failed. If False, negatives are passed through to
        aggregation (cost may still be NaN if obs_vec contains zeros).

    Returns
    -------
    cost_grid : np.ndarray, shape (n1, n2)
        NRMSRE cost per cell; NaN for failed cells.
    model_grid : np.ndarray, shape (n1, n2, n_targets)
        Aggregated model target vector per cell; NaN for failed cells.
    """
    tail = slice(-avg_window, None)
    state_das = _build_state_das(
        scan_results, bin_definitions,
        time_collapser=lambda da: da.isel(time=tail).mean('time'),
    )
    d_e_scalar, d_e_da = _resolve_de(
        scan_results, set(b['type'] for b in bin_definitions)
    )

    return _iterate_cost_grid(
        state_das,
        n1=len(scan_results[dim1_name]),
        n2=len(scan_results[dim2_name]),
        dim1_name=dim1_name, dim2_name=dim2_name,
        bin_definitions=bin_definitions, obs_vec=obs_vec,
        phyto_esd=phyto_esd, zoo_esd=zoo_esd,
        d_e_scalar=d_e_scalar, d_e_da=d_e_da,
        stable_mask=None,
        neg_tolerance=0.0 if reject_negative else np.inf,
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
    """Post-process an xso.parscans stability scan into a cost grid.

    For each cell, each target type's XSO output variable (see
    TARGET_EXTRACTORS) is evaluated at the fsolve steady state (time=-1),
    aggregated into the target vector via `aggregate_model_to_targets`,
    and scored against `obs_vec` using NRMSRE.

    A cell becomes NaN in the cost grid if:
      - `require_stable=True` and the cell's stability label is not
        'stable'
      - any extracted value is < -neg_tolerance (treated as a true
        negative rather than floating-point noise)
      - any extracted value is NaN (fsolve failed to converge)

    Small negative values in [-neg_tolerance, 0) are clipped to 0 before
    aggregation. This is needed because fsolve reports roots up to its
    tolerance, which can produce state variables a few machine-epsilons
    below zero at true-zero equilibria.

    Requires an XSO version where the NumericalStabilitySolver evaluates
    flux functions at the steady state (rather than zeroing them).
    Earlier versions return 0 for all flux variables, which would make
    'pp' and 'export' targets silently yield zero model values.

    Parameters
    ----------
    stability_results : xarray.Dataset
        Output from `run_xso_stabilityscan`. Must contain each XSO
        variable named in TARGET_EXTRACTORS for the types used, plus a
        'stability' variable and (if any 'export' target is present)
        'Inflow__de'.
    phyto_esd, zoo_esd : array-like
        Size-class centers (µm ESD).
    obs_vec : np.ndarray, shape (n_targets,)
        Observation target vector.
    bin_definitions : list of dict
        Target bin definitions (must match obs_vec ordering).
    dim1_name, dim2_name : str
        Names of the two scan dimensions in `stability_results`.
    neg_tolerance : float
        Magnitude below which negative values are treated as fsolve noise
        and clipped to 0. Values more negative than -neg_tolerance flag
        the cell as failed.
    require_stable : bool
        If True (default), cells where the stability label is not
        'stable' are flagged as failed. If False, unstable equilibria
        are still scored (useful for exploring cost-landscape structure
        beyond the stable region).

    Returns
    -------
    cost_grid : np.ndarray, shape (n1, n2)
        NRMSRE cost per cell; NaN for failed cells.
    model_grid : np.ndarray, shape (n1, n2, n_targets)
        Aggregated model target vector per cell; NaN for failed cells.
    """
    state_das = _build_state_das(
        stability_results, bin_definitions,
        time_collapser=lambda da: da.isel(time=-1),
    )
    d_e_scalar, _ = _resolve_de(
        stability_results, set(b['type'] for b in bin_definitions)
    )
    stable_mask = (stability_results['stability'] == 'stable'
                   if require_stable else None)

    return _iterate_cost_grid(
        state_das,
        n1=len(stability_results[dim1_name]),
        n2=len(stability_results[dim2_name]),
        dim1_name=dim1_name, dim2_name=dim2_name,
        bin_definitions=bin_definitions, obs_vec=obs_vec,
        phyto_esd=phyto_esd, zoo_esd=zoo_esd,
        d_e_scalar=d_e_scalar, d_e_da=None,
        stable_mask=stable_mask,
        neg_tolerance=neg_tolerance,
        clip_small_negatives=True,
    )