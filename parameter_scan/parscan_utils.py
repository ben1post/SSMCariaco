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
def old_compute_cost_nrmsre(model_vec, obs_vec):
    """
    Normalized Root Mean Square Relative Error.

        cost = sqrt( (1/N) * Σ ((model_i - obs_i) / obs_i)^2 )

    Dimensionless — all targets weighted equally regardless of magnitude.
    Cost ~0.3 means "on average ~30% relative error across targets."
    """
    rel_errors = (model_vec - obs_vec) / obs_vec
    return np.sqrt(np.mean(rel_errors ** 2))
    

def compute_cost_nrmsre(model_vec, obs_vec):
    """
    Log-space RMSE between model and observations.

        cost = sqrt( (1/N) * Σ (log10(model_i) - log10(obs_i))^2 )

    Dimensionless. Symmetric in factor-above vs factor-below obs
    (10× too high and 10× too low both cost 1.0). Diverges for
    model_i → 0, which penalizes ecologically-meaningless extinction
    solutions that linear-relative-error costs reward.

    Non-positive model values are floored at a small epsilon to keep
    the log finite while still giving them a very large cost.
    """
    eps = 1e-12
    model_safe = np.where(np.asarray(model_vec) > eps, model_vec, eps)
    log_errors = np.log10(model_safe) - np.log10(obs_vec)
    return np.sqrt(np.mean(log_errors ** 2))


def compute_cost_relative_spectrum(model_vec, obs_vec, bin_definitions,
                                   type_filter='phyto'):
    """
    Euclidean distance between relative-composition vectors of one target type.

    Restricts attention to targets whose ``type`` field equals ``type_filter``
    (default: ``'phyto'``), normalises the model and obs sub-vectors to sum
    to 1, and returns the Euclidean distance between them. Insensitive to
    total biomass — measures *spectrum slope* only.

        cost = || m / sum(m)  -  o / sum(o) ||_2

    Range: 0 (perfect composition match) to sqrt(2) (orthogonal
    compositions, e.g. all-Pico vs all-Micro). Returns NaN if either the
    model or the obs sub-vector sums to a non-positive value.

    Intended as a *secondary* cost alongside ``compute_cost_nrmsre``: the
    NRMSRE captures absolute magnitudes across all targets, while this
    function isolates the size-spectrum slope of a single target type so
    the trade-off between magnitude fit and slope fit can be inspected.

    Parameters
    ----------
    model_vec, obs_vec : array-like, shape (n_targets,)
        Full target vectors as built by ``aggregate_model_to_targets`` /
        ``load_cariaco_targets``. Same length and target ordering.
    bin_definitions : list of dict
        Bin definitions used to build the target vectors. Required so the
        function knows which entries correspond to ``type_filter``.
    type_filter : str, optional
        Target type whose relative composition is scored. Default
        ``'phyto'`` (Pico / Nano / Micro slope).

    Returns
    -------
    cost : float
        Euclidean distance between normalised composition vectors, or NaN
        if model or obs sums to a non-positive value.
    """
    idx = [i for i, b in enumerate(bin_definitions)
           if b['type'] == type_filter]
    if len(idx) < 2:
        raise ValueError(
            f"compute_cost_relative_spectrum needs at least 2 targets of "
            f"type '{type_filter}' in bin_definitions, found {len(idx)}."
        )
    m = np.asarray(model_vec, dtype=float)[idx]
    o = np.asarray(obs_vec, dtype=float)[idx]
    m_sum, o_sum = m.sum(), o.sum()
    if not np.isfinite(m_sum) or not np.isfinite(o_sum):
        return np.nan
    if m_sum <= 0 or o_sum <= 0:
        return np.nan
    return float(np.linalg.norm(m / m_sum - o / o_sum))


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


def _adaptive_time_collapser(da, avg_window=None):
    """Collapse a DataArray's 'time' dim according to its length.

    - no 'time' dim → pass through unchanged
    - 'time' length 1 → isel(time=0) (e.g. output of avg_tail postprocess)
    - 'time' length 2 → isel(time=-1) (stability scan: [initial, steady])
    - 'time' length > 2 → tail-mean over `avg_window` (raw IVP scan)

    Raises
    ------
    ValueError
        If 'time' length > 2 and `avg_window` is None.
    """
    if 'time' not in da.dims:
        return da

    n = da.sizes['time']
    if n == 1:
        return da.isel(time=0)
    if n == 2:
        return da.isel(time=-1)

    if avg_window is None:
        raise ValueError(
            f"'{da.name}' has time dim of length {n} (> 2) but avg_window "
            f"was not provided. Either pre-collapse the scan output (e.g. "
            f"with xso.parscans.avg_tail) or pass avg_window explicitly."
        )
    return da.isel(time=slice(-avg_window, None)).mean('time')

    
def _build_state_das(results_ds, bin_definitions, avg_window=None):
    """Extract target-type DataArrays from scan output, collapsing time.

    Walks over the unique target types in `bin_definitions`, looks each one
    up in TARGET_EXTRACTORS to find its XSO output variable name, and
    applies `_adaptive_time_collapser` so the caller doesn't need to know
    whether the input is a raw IVP scan, a post-processed IVP scan, or a
    stability scan.
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
        state_das[t] = _adaptive_time_collapser(results_ds[var_name], avg_window)
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
# 2D COST GRID 
# =============================================================================
def compute_cost_grid(
    results_ds,
    phyto_esd, zoo_esd,
    obs_vec,
    bin_definitions,
    dim1_name, dim2_name,
    *,
    avg_window=None,
    neg_tolerance=0.0,
    clip_small_negatives=False,
    require_stable=False,
):
    """Post-process a 2D XSO parameter-scan Dataset into a cost grid.

    Handles IVP, post-processed IVP, and stability scans uniformly —
    the time dim is collapsed adaptively based on its length (see
    `_adaptive_time_collapser`). Variables without a time dim are
    passed through. For each cell, target values are aggregated via
    `aggregate_model_to_targets` and scored against `obs_vec` using
    NRMSRE.

    Typical usage
    -------------
    Post-processed IVP (time=1, e.g. after `avg_tail` hook)::

        compute_cost_grid(scan_results, phyto_esd, zoo_esd, obs_vec,
                          bin_defs, P1, P2)

    Raw IVP (time=5000)::

        compute_cost_grid(scan_results, ..., avg_window=1000)

    Stability scan (time=2)::

        compute_cost_grid(stability_results, ...,
                          neg_tolerance=1e-6,
                          clip_small_negatives=True,
                          require_stable=True)

    Parameters
    ----------
    results_ds : xarray.Dataset
        Output from `run_xso_parscan` or `run_xso_stabilityscan`. Must
        contain each XSO variable named in TARGET_EXTRACTORS for the
        target types used. If any 'export' target is present, must also
        contain 'Inflow__de'. If `require_stable=True`, must also contain
        'stability'.
    phyto_esd, zoo_esd : array-like
        Size-class centers (µm ESD).
    obs_vec : np.ndarray, shape (n_targets,)
    bin_definitions : list of dict
    dim1_name, dim2_name : str
        Names of the two scan dimensions in `results_ds`.
    avg_window : int or None, optional
        Number of final time steps to average when the time dim has
        length > 2. Required only in that case; silently ignored when
        time is absent, length 1, or length 2.
    neg_tolerance : float, optional
        Values more negative than -neg_tolerance flag the cell as failed
        (cost = NaN). Default 0.0 → any negative fails. Use a small
        positive value (e.g. 1e-6) for fsolve-based stability scans to
        tolerate floating-point noise around zero.
    clip_small_negatives : bool, optional
        If True, values in [-neg_tolerance, 0) are clipped to 0 before
        aggregation. Typical companion to a nonzero `neg_tolerance`.
    require_stable : bool, optional
        If True, cells whose 'stability' label is not 'stable' are
        flagged as failed. Requires a 'stability' variable in `results_ds`.

    Returns
    -------
    cost_grid : np.ndarray, shape (n1, n2)
    model_grid : np.ndarray, shape (n1, n2, n_targets)
        Both NaN for failed cells.
    """
    state_das = _build_state_das(results_ds, bin_definitions, avg_window)
    d_e_scalar, d_e_da = _resolve_de(
        results_ds, set(b['type'] for b in bin_definitions)
    )

    if require_stable:
        if 'stability' not in results_ds.variables:
            raise ValueError(
                "require_stable=True, but 'stability' is not in results_ds."
            )
        stable_mask = results_ds['stability'] == 'stable'
    else:
        stable_mask = None

    return _iterate_cost_grid(
        state_das,
        n1=len(results_ds[dim1_name]),
        n2=len(results_ds[dim2_name]),
        dim1_name=dim1_name, dim2_name=dim2_name,
        bin_definitions=bin_definitions, obs_vec=obs_vec,
        phyto_esd=phyto_esd, zoo_esd=zoo_esd,
        d_e_scalar=d_e_scalar, d_e_da=d_e_da,
        stable_mask=stable_mask,
        neg_tolerance=neg_tolerance,
        clip_small_negatives=clip_small_negatives,
    )


def compute_spectrum_cost_grid(model_grid, obs_vec, bin_definitions,
                               type_filter='phyto'):
    """
    Reduce a (n1, n2, n_targets) model_grid to a (n1, n2) spectrum-only
    cost grid, scoring each cell's *relative composition* against obs.

    Operates on the post-processed ``model_grid`` produced by
    ``compute_cost_grid`` — does not re-run the scan or read the parent
    ``results_ds``. Cells that already failed in the parent scan
    (NaN / non-finite entries in the per-cell model_vec) are propagated
    as NaN.

    The same NaN footprint as the parent NRMSRE ``cost_grid`` is
    therefore preserved: cells masked-out upstream stay masked-out here.

    Parameters
    ----------
    model_grid : np.ndarray, shape (n1, n2, n_targets)
        Per-cell aggregated target vectors from ``compute_cost_grid``.
    obs_vec : np.ndarray, shape (n_targets,)
    bin_definitions : list of dict
        Same definitions used to build ``model_grid``.
    type_filter : str, optional
        Target type whose relative composition is scored, passed through
        to ``compute_cost_relative_spectrum``. Default ``'phyto'``.

    Returns
    -------
    spectrum_cost_grid : np.ndarray, shape (n1, n2)
        Relative-composition distance per cell; NaN where the parent
        model_vec contained any NaN/inf or summed to <= 0 over the
        filtered targets.

    See Also
    --------
    compute_cost_relative_spectrum : per-cell scoring used internally
    compute_cost_grid : produces the ``model_grid`` consumed here
    """
    model_grid = np.asarray(model_grid)
    if model_grid.ndim != 3:
        raise ValueError(
            f"model_grid must be 3-D (n1, n2, n_targets); got shape "
            f"{model_grid.shape}."
        )
    n1, n2, _ = model_grid.shape
    grid = np.full((n1, n2), np.nan)
    for i in range(n1):
        for j in range(n2):
            mv = model_grid[i, j, :]
            if not np.all(np.isfinite(mv)):
                continue
            grid[i, j] = compute_cost_relative_spectrum(
                mv, obs_vec, bin_definitions, type_filter=type_filter,
            )
    return grid


# =============================================================================
# PHYTO SIZE-SPECTRUM METRICS
# =============================================================================
# Cariaco bin geomeans matching the model-resolved size range (0.5-2 / 2-20 /
# 20-200 µm). The Pico geomean = sqrt(0.5×2) = 1.0 µm deliberately departs
# from canonical Sieburth (0.2-2 µm → geomean 0.63) because: (a) Cariaco Pico
# is dominated by Synechococcus + picoeukaryotes (≥ 0.5 µm) per Lorenzoni
# 2015 §3.1 with Prochlorococcus rare in coastal upwelling; (b) the MS3
# model resolves 0.5-200 µm; (c) 1-1 consistency between model, obs metric,
# and analytical derivation integration limits. Updated 2026-05-13;
# previous value was [0.63, 6.3, 63.0]. Mirrors `depth_profile_data.r`
# `size_centroid` geomean weights. Override `bin_geomeans` to test
# alternative conventions.
CARIACO_PHYTO_BIN_GEOMEANS = np.array([1.0, 6.3, 63.0])


def compute_phyto_spectrum_metrics(model_vec, bin_definitions,
                                   bin_geomeans=None):
    """Compute Cariaco-side phyto size-spectrum metrics from a model target vector.

    Parallel to ``compute_cost_relative_spectrum`` but returns the metrics
    themselves (centroid, Shannon evenness, 2-point Pico→Micro slope) rather
    than a fit cost. The formulas match the Cariaco observation pipeline
    (R-side ``depth_profile_data.r``, post-2026-05-12 Sathyendranath C:Chl
    refactor) so the model output is directly comparable to the obs envelope
    summarised in ``cariaco_monthly_euphotic_dynamic.csv``.

    Restricts attention to bins with ``type='phyto'`` — the three Cariaco
    Pico / Nano / Micro bins as defined in ``TARGET_BIN_DEFINITIONS``. The
    canonical ordering (smallest-first) determines which bin is which:
    first entry = Pico (slope's lower endpoint), last entry = Micro (upper).

    Formulas (biomass-based, matching depth_profile_data.r):

        centroid    = Σ_i p_i · log10(ESD_i)        — biomass-weighted log ESD
        shannon     = -Σ_i p_i · ln(p_i)            — natural log; max ln(3)
        nbss_slope  = (log10(B_micro) - log10(B_pico))
                    / (log10(ESD_micro) - log10(ESD_pico))

    Parameters
    ----------
    model_vec : array-like, shape (n_targets,)
        Per-bin biomass values as produced by ``aggregate_model_to_targets``.
    bin_definitions : list of dict
        Same definitions used to build ``model_vec``. Only entries with
        ``type='phyto'`` are used here; their order in the list determines
        the fraction order and the slope endpoints.
    bin_geomeans : array-like, shape (n_phyto,) or None, optional
        Geometric-mean ESD (µm) for each phyto bin, in the same order as
        the phyto entries of ``bin_definitions``. Defaults to
        ``CARIACO_PHYTO_BIN_GEOMEANS`` (= [1.0, 6.3, 63.0], model-resolved;
        updated 2026-05-13 from the previous Sieburth [0.63, 6.3, 63.0]).

    Returns
    -------
    metrics : dict
        Keys ``'centroid'``, ``'shannon'``, ``'nbss_slope'``, ``'fractions'``.
        Scalar floats except ``'fractions'`` which is a 1-D np.ndarray in
        phyto-bin order. All return NaN (and fractions all-NaN) if total
        phyto biomass is non-positive or any phyto bin is non-finite. The
        slope alone returns NaN if either slope endpoint (first or last
        bin biomass) is non-positive.
    """
    if bin_geomeans is None:
        bin_geomeans = CARIACO_PHYTO_BIN_GEOMEANS
    bin_geomeans = np.asarray(bin_geomeans, dtype=float)

    phyto_idx = [i for i, b in enumerate(bin_definitions)
                 if b['type'] == 'phyto']
    if len(phyto_idx) != len(bin_geomeans):
        raise ValueError(
            f"compute_phyto_spectrum_metrics: bin_definitions has "
            f"{len(phyto_idx)} phyto entries but bin_geomeans has "
            f"{len(bin_geomeans)}. Expected matching counts."
        )
    if len(phyto_idx) < 2:
        raise ValueError(
            f"compute_phyto_spectrum_metrics needs at least 2 phyto bins "
            f"for the slope endpoint pair, got {len(phyto_idx)}."
        )

    biomass = np.asarray(model_vec, dtype=float)[phyto_idx]

    nan_result = dict(
        centroid=np.nan,
        shannon=np.nan,
        nbss_slope=np.nan,
        fractions=np.full(len(phyto_idx), np.nan),
    )

    if not np.all(np.isfinite(biomass)):
        return nan_result
    total = biomass.sum()
    if total <= 0:
        return nan_result

    fractions = biomass / total
    log_esd = np.log10(bin_geomeans)

    centroid = float(np.dot(fractions, log_esd))

    with np.errstate(invalid='ignore', divide='ignore'):
        shannon_terms = np.where(
            fractions > 0, fractions * np.log(fractions), 0.0
        )
    shannon = float(-shannon_terms.sum())

    B_pico = biomass[0]
    B_micro = biomass[-1]
    if B_pico > 0 and B_micro > 0:
        nbss_slope = float(
            (np.log10(B_micro) - np.log10(B_pico))
            / (log_esd[-1] - log_esd[0])
        )
    else:
        nbss_slope = np.nan

    return dict(
        centroid=centroid,
        shannon=shannon,
        nbss_slope=nbss_slope,
        fractions=fractions,
    )


def compute_phyto_spectrum_metrics_grid(model_grid, bin_definitions,
                                        bin_geomeans=None):
    """Reduce a (n1, n2, n_targets) model_grid to per-cell phyto-metric grids.

    Parallel to ``compute_spectrum_cost_grid`` but returns the metrics
    (centroid, Shannon, nbss_slope) rather than a fit cost. Iterates over
    cells once and returns all three metrics — cheaper than calling
    ``compute_phyto_spectrum_metrics`` three times if the caller wants
    more than one metric.

    Operates on the post-processed ``model_grid`` from ``compute_cost_grid``
    — does not re-run the scan. NaN footprint is the parent scan's plus any
    cells where total phyto biomass is non-positive.

    1-D scans: pass a model_grid of shape ``(n1, 1, n_targets)``; the
    returned grids will have shape ``(n1, 1)``. Or call
    ``compute_phyto_spectrum_metrics`` directly in a comprehension.

    Parameters
    ----------
    model_grid : np.ndarray, shape (n1, n2, n_targets)
    bin_definitions : list of dict
    bin_geomeans : array-like or None, optional

    Returns
    -------
    grids : dict[str, np.ndarray]
        Keys ``'centroid'``, ``'shannon'``, ``'nbss_slope'``. Each value is
        a ``(n1, n2)`` ``np.ndarray``; NaN for failed cells.
    """
    model_grid = np.asarray(model_grid)
    if model_grid.ndim != 3:
        raise ValueError(
            f"model_grid must be 3-D (n1, n2, n_targets); got shape "
            f"{model_grid.shape}."
        )
    n1, n2, _ = model_grid.shape
    grids = {
        'centroid':   np.full((n1, n2), np.nan),
        'shannon':    np.full((n1, n2), np.nan),
        'nbss_slope': np.full((n1, n2), np.nan),
    }
    for i in range(n1):
        for j in range(n2):
            metrics = compute_phyto_spectrum_metrics(
                model_grid[i, j, :], bin_definitions, bin_geomeans,
            )
            for k in grids:
                grids[k][i, j] = metrics[k]
    return grids


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




def run_single_point(model, model_setup, scan_params, fixed_overrides=None):
    """Run a single IVP simulation at a specific parameter combination.

    Useful for re-running the model at best-fit parameters to obtain a full
    time series for plotting, after a scan whose output may have been
    time-collapsed by a postprocess hook (or a stability scan, which never
    has a full time series to begin with).

    Note: the parscan postprocess hook is NOT applied here — a direct
    ``xsimlab.run()`` call always returns the full time series, regardless
    of what ``run_xso_parscan`` was configured with.

    Parameters
    ----------
    model : xsimlab.Model
        The XSO model (e.g. `cariaco_ssm_setup.model`).
    model_setup : xarray.Dataset
        The model setup (e.g. `model_setup_slim`). Governs which variables
        are stored in the output.
    scan_params : dict
        Parameter overrides for this run, e.g.
        ``{P1_NAME: best['val1'], P2_NAME: best['val2']}``.
    fixed_overrides : dict or None, optional
        Additional overrides (e.g. regime-specific forcing values).

    Returns
    -------
    xarray.Dataset
        Full time-series output of the single run.
    """
    overrides = dict(scan_params)
    if fixed_overrides:
        overrides.update(fixed_overrides)
    with model:
        return model_setup.xsimlab.update_vars(input_vars=overrides).xsimlab.run()