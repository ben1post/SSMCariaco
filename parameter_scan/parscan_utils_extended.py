"""
parscan_utils_extended.py — extends parscan_utils.py with the per-target
diagnostic patterns we developed for the MS3 baseline (2026-06-14).

Reuses from parscan_utils:
- get_log_bin_edges, get_fraction_in_range
- CARIACO_PHYTO_BIN_GEOMEANS
- run_single_point (re-exported for convenience)

Adds (this file):
- avg_tail_stats(ds, avg_window=1000) -> parscan postprocess hook: avg_tail
  PLUS realised-stability scalars (cv_sumP, cv_sumZ, cv_N, tail_has_nan).
  Re-export from the model setup module so the worker finds it by name.
- compute_sieburth_weights(phyto_esd) -> (3, n_phyto) overlap-weight matrix
- extract_scan_metrics_1d(scan, axis_name, forcing) -> dict of per-variable arrays
- extract_scan_metrics_2d(scan, axis_name_1, axis_name_2, forcing) -> dict
- build_obs_refs(monthly_df) -> dict of per-target obs medians
- joint_error(metrics, obs_refs, weights=None, targets=...) -> per-point L2 error
- FLAG_THRESHOLDS + compute_flags(metrics, obs_refs) -> admissibility / shape
  flags (annotate, never filter). See the flag layer section below.
- print_scan_table_1d(metrics, obs_refs, header, axis_label)  [+ CV / flag cols]
- print_scan_table_2d(metrics, obs_refs, header, axis_labels, n_top=5,
                      weights_eq=..., weights_zw=...)
- plot_scan_panels_1d(metrics, obs_refs, header, axis_label)  [+ CV panel / flags]
- plot_2d_heatmaps(metrics, obs_refs, header, axis_labels)

Stability layer (2026-06-20): the harness annotates and RANKS candidate fits,
it never filters them. CV(ΣP) comes from avg_tail_stats (computed in the
parscan worker from the tail before reduction); flags surface caveats so a
good-but-flagged fit still reaches the user. Harness F_N sweeps should run
with postprocess_name='avg_tail_stats' AND
solver_kwargs={'instability_neg_threshold': -1e-3} so oscillatory bloom cells
run to completion and yield a real CV instead of being NaN-padded.

API conventions:
- "metrics" is a dict of numpy arrays keyed by variable name (mcs, P_pico,
  P_nano, P_micro, sumP, Z_gt200, Z_gt500, sumZ, N, D, PP, Export, plus 'axis'
  for 1D or 'axis_1' / 'axis_2' for 2D). Shape is (n_axis,) for 1D scans and
  (n_axis_1, n_axis_2) for 2D.
- "obs_refs" is a dict of scalars (or np.nan where obs is unavailable) keyed
  by the same variable names.
- Extraction functions use xarray named-dim operations throughout — they are
  agnostic to the underlying dim ordering in the parscan output.
- The extraction functions expect the slim-output set we defined for
  baseline_r0_setups: state vars + Growth__uptake_value + DetritusSink__sinking_value.
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from parscan_utils import (
    get_log_bin_edges,
    get_fraction_in_range,
    CARIACO_PHYTO_BIN_GEOMEANS,
    run_single_point,                # re-exported for convenience
)


# =============================================================================
# Realised-stability postprocess hook + helper
# =============================================================================
def _tail_cv(arr):
    """(cv, has_nan) for a 1-D tail series, nan-aware.

    cv = std / |mean| over the finite part of the tail; np.nan if there are
    no finite points or the mean is zero. has_nan is True if any non-finite
    value appears anywhere in the tail (catches instability-event NaN-padding
    and NaN-sentinel failed cells).
    """
    a = np.asarray(arr, dtype=float).ravel()
    has_nan = bool(np.any(~np.isfinite(a)))
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan, has_nan
    mean = a.mean()
    if mean == 0 or not np.isfinite(mean):
        return np.nan, has_nan
    return float(a.std() / abs(mean)), has_nan


def avg_tail_stats(ds, avg_window=1000, phyto_dim='phyto', zoo_dim='zoo'):
    """Stability-aware drop-in for xso.parscans.avg_tail.

    Reduces every time-dimensioned variable to its mean over the last
    ``avg_window`` time steps (``time`` kept as a length-1 coordinate;
    no-time variables passed through), so all downstream extraction is
    byte-identical to an avg_tail run. In ADDITION it computes realised-
    stability scalars from the tail window BEFORE reduction and merges them
    in as no-time variables (so run_xso_parscan concatenates them along the
    scan axis):

      - cv_sumP : CV (std/|mean|) of total phyto biomass ΣP(t) over the tail
      - cv_sumZ : CV of total zoo biomass ΣZ(t) over the tail
      - cv_N    : CV of dissolved N(t) over the tail
      - tail_has_nan : 1.0 if any NaN appears in the ΣP / ΣZ / N tail
                       (instability-event padding or a NaN sentinel), else 0.0

    Tail statistics are nan-aware (ΣP/ΣZ are summed with skipna=False so a
    NaN-padded step propagates to the total and is detected): a partially
    NaN-padded tail still yields a CV from its finite head while
    tail_has_nan flags it. Re-export this from the model setup module so the
    parscan worker can resolve it by name (postprocess_name='avg_tail_stats').
    """
    if 'time' not in ds.dims:
        return ds

    final_t = ds['time'].values[-1]
    tail = ds.isel(time=slice(-avg_window, None))

    # --- realised-stability scalars (before reduction) ---
    stat_vars = {}
    any_nan = False
    if 'Phytoplankton__biomass' in ds:
        sumP_t = tail['Phytoplankton__biomass'].sum(dim=phyto_dim, skipna=False)
        cv, hn = _tail_cv(sumP_t.values); stat_vars['cv_sumP'] = cv; any_nan |= hn
    if 'Zooplankton__biomass' in ds:
        sumZ_t = tail['Zooplankton__biomass'].sum(dim=zoo_dim, skipna=False)
        cv, hn = _tail_cv(sumZ_t.values); stat_vars['cv_sumZ'] = cv; any_nan |= hn
    if 'Nutrient__value' in ds:
        cv, hn = _tail_cv(tail['Nutrient__value'].values); stat_vars['cv_N'] = cv
        any_nan |= hn
    stat_vars['tail_has_nan'] = 1.0 if any_nan else 0.0

    # --- avg_tail reduction (identical to xso.parscans.avg_tail) ---
    vars_with_time = [v for v in ds.data_vars if 'time' in ds[v].dims]
    vars_without_time = [v for v in ds.data_vars if 'time' not in ds[v].dims]

    if vars_with_time:
        reduced = (ds[vars_with_time].isel(time=slice(-avg_window, None))
                   .mean('time', keep_attrs=True)
                   .expand_dims({'time': [final_t]}))
    else:
        reduced = xr.Dataset()

    out = xr.merge([reduced, ds[vars_without_time]]) if vars_without_time else reduced

    for k, v in stat_vars.items():
        out[k] = xr.DataArray(np.float64(v))

    out.attrs = ds.attrs
    return out


# =============================================================================
# Sieburth bin conventions (defaults match parscan_utils / cariaco_obs)
# =============================================================================
DEFAULT_SIEBURTH_RANGES = [(0.0, 2.0), (2.0, 20.0), (20.0, np.inf)]  # Pico, Nano, Micro
DEFAULT_SIEBURTH_LABELS = ['Pico', 'Nano', 'Micro']
DEFAULT_ZOO_THRESHOLDS = (200.0, 500.0)   # cumulative >threshold bands

# Size-spectrum targets we score against obs by default
DEFAULT_TARGETS = ['mcs', 'P_pico', 'P_nano', 'P_micro',
                   'Z_gt200', 'Z_gt500']

# Flag thresholds for compute_flags (annotate, never filter). Each is a single
# explicit, tunable value — provenance in the comments.
FLAG_THRESHOLDS = dict(
    cv_warn=0.05,        # CV(ΣP) above this = mild oscillation (survey: stable
                         #   construct ~1e-4, limit cycle ~0.4-0.8 → 0.05 cleanly
                         #   separates a converged SS from an oscillating one)
    cv_strong=0.20,      # CV(ΣP) above this = strong limit cycle
    n_ceiling=100.0,     # N > this (mmol N m-3) = runaway (Benny 2026-06-20;
                         #   obs euphotic N is O(1), model ~0.5 at high F_N)
    frac_floor=0.02,     # any Sieburth bin fraction below this = coexistence loss
                         #   (Pico-monoculture or Nano/Micro extinction)
    sumP_floor=1e-6,     # ΣP below this (mmol N m-3) = collapsed system
    mcs_desc_tol=0.02,   # non-decreasing check: ignore relative dips ≤ this
                         #   fraction as numerical jitter (supervisor bar = no
                         #   decrease in mcs with rising F_N)
    largeZ_obs_frac=0.25,  # Z>200 below this × obs Z>200 across the whole range
                           #   = large-Z pool too thin for the fish lever
                           #   (project_fish_largez precondition)
    comp_tol=0.15,       # composition flags: relative tolerance on the F_N-local
                         #   obs fraction (nano_over fires if f_nano_model >
                         #   f_nano_obs(F_N)·(1+comp_tol); micro_short symmetric)
)


# =============================================================================
# Sieburth weights — fractional log-overlap, model classes → 3 obs bins
# =============================================================================
def compute_sieburth_weights(phyto_esd,
                             sieburth_ranges=DEFAULT_SIEBURTH_RANGES):
    """(n_bins, n_phyto) weight matrix W[k, i] = fraction of phyto class i's
    log-extent that falls inside Sieburth bin k. Aggregating model bin biomass
    via `bins = P @ W.T` matches the obs side (which is itself bin biomass)."""
    phyto_esd = np.asarray(phyto_esd)
    p_edges = get_log_bin_edges(phyto_esd)
    return np.array([
        [get_fraction_in_range(p_edges[i], p_edges[i + 1], lo, hi)
         for i in range(len(phyto_esd))]
        for (lo, hi) in sieburth_ranges
    ])


# =============================================================================
# Extraction — 1D and 2D parscan output -> per-variable metrics dict
# =============================================================================
def _build_geomeans_da(bin_geomeans, bin_labels):
    return xr.DataArray(np.asarray(bin_geomeans), dims='bin',
                        coords={'bin': list(bin_labels)})


def _build_W_phyto_da(phyto_esd, sieburth_ranges, bin_labels,
                      phyto_dim_name='phyto'):
    W = compute_sieburth_weights(phyto_esd, sieburth_ranges)
    return xr.DataArray(W, dims=('bin', phyto_dim_name),
                        coords={'bin': list(bin_labels),
                                phyto_dim_name: phyto_esd})


def _extract_common(scan, forcing,
                    phyto_dim='phyto', zoo_dim='zoo',
                    bin_geomeans=CARIACO_PHYTO_BIN_GEOMEANS,
                    bin_labels=DEFAULT_SIEBURTH_LABELS,
                    sieburth_ranges=DEFAULT_SIEBURTH_RANGES,
                    zoo_thresholds=DEFAULT_ZOO_THRESHOLDS):
    """Compute the derived metric DataArrays from a parscan dataset.

    Returns a dict of xarray DataArrays keyed by variable name. The dataset's
    parscan axes are preserved (one axis for 1D scan, two for 2D); the
    `phyto_dim` / `zoo_dim` dims are reduced.
    """
    # Coord arrays from the scan dataset
    phyto_esd = scan[phyto_dim].values

    # Sieburth bin weights + bin geomeans as labelled DataArrays
    W_da        = _build_W_phyto_da(phyto_esd, sieburth_ranges, bin_labels, phyto_dim)
    geomeans_da = _build_geomeans_da(bin_geomeans, bin_labels)

    # State vars — collapse time (length 1 after avg_tail; pass-through if absent)
    def _squeeze_time(da):
        return da.squeeze('time') if 'time' in da.dims else da

    P_da    = _squeeze_time(scan['Phytoplankton__biomass'])
    Z_da    = _squeeze_time(scan['Zooplankton__biomass'])
    N_da    = _squeeze_time(scan['Nutrient__value'])
    D_da    = _squeeze_time(scan['Detritus__value'])
    PPpc_da = _squeeze_time(scan['Growth__uptake_value'])
    expv_da = _squeeze_time(scan['DetritusSink__sinking_value'])

    # Sieburth bin biomass via xarray multiplication (aligns by phyto coord)
    bins_da = (W_da * P_da).sum(dim=phyto_dim)
    P_pico_da  = bins_da.sel(bin='Pico')
    P_nano_da  = bins_da.sel(bin='Nano')
    P_micro_da = bins_da.sel(bin='Micro')

    # mcs from bin fractions × log10(geomeans)
    frac_da = bins_da / bins_da.sum(dim='bin').clip(min=1e-30)
    mcs_da  = 10.0 ** ((frac_da * np.log10(geomeans_da)).sum(dim='bin'))

    # Totals
    sumP_da = P_da.sum(dim=phyto_dim)
    sumZ_da = Z_da.sum(dim=zoo_dim)

    # Cumulative >threshold zoo bands
    z_coord = scan[zoo_dim]
    Z_band_das = {}
    for thr in zoo_thresholds:
        key = f'Z_gt{int(thr)}'
        Z_band_das[key] = Z_da.where(z_coord > thr, 0).sum(dim=zoo_dim)

    # PP (total uptake) and Export (areal flux)
    PP_da     = PPpc_da.sum(dim=phyto_dim)
    d_e       = float(forcing['Inflow__de'])
    Export_da = expv_da * d_e

    das = dict(
        mcs=mcs_da, P_pico=P_pico_da, P_nano=P_nano_da, P_micro=P_micro_da,
        sumP=sumP_da, sumZ=sumZ_da,
        N=N_da, D=D_da, PP=PP_da, Export=Export_da,
        **Z_band_das,
    )

    # Realised-stability scalars — present only when avg_tail_stats was the
    # postprocess. Passed through verbatim (axis dim only); absent otherwise.
    for k in ('cv_sumP', 'cv_sumZ', 'cv_N', 'tail_has_nan'):
        if k in scan:
            sda = scan[k]
            das[k] = sda.squeeze('time') if 'time' in sda.dims else sda

    return das


def extract_scan_metrics_1d(scan, axis_name, forcing,
                            phyto_dim='phyto', zoo_dim='zoo',
                            bin_geomeans=CARIACO_PHYTO_BIN_GEOMEANS,
                            bin_labels=DEFAULT_SIEBURTH_LABELS,
                            sieburth_ranges=DEFAULT_SIEBURTH_RANGES,
                            zoo_thresholds=DEFAULT_ZOO_THRESHOLDS):
    """Pull standard diagnostic arrays from a 1D parscan dataset.

    Returns a dict of numpy arrays (length n_axis) sorted by the axis coord.
    Keys: 'axis', 'mcs', 'P_pico', 'P_nano', 'P_micro', 'sumP', 'sumZ',
    'Z_gt200', 'Z_gt500', 'N', 'D', 'PP', 'Export'.
    """
    das = _extract_common(scan, forcing, phyto_dim, zoo_dim, bin_geomeans,
                          bin_labels, sieburth_ranges, zoo_thresholds)
    axis_da = scan[axis_name]
    order = np.argsort(axis_da.values)
    axis = axis_da.values[order]

    out = {'axis': axis}
    for k, da in das.items():
        out[k] = da.transpose(axis_name).values[order]
    return out


def extract_scan_metrics_2d(scan, axis_name_1, axis_name_2, forcing,
                            phyto_dim='phyto', zoo_dim='zoo',
                            bin_geomeans=CARIACO_PHYTO_BIN_GEOMEANS,
                            bin_labels=DEFAULT_SIEBURTH_LABELS,
                            sieburth_ranges=DEFAULT_SIEBURTH_RANGES,
                            zoo_thresholds=DEFAULT_ZOO_THRESHOLDS):
    """Pull standard diagnostic arrays from a 2D parscan dataset.

    Returns a dict; per-variable arrays have shape (n_axis_1, n_axis_2),
    sorted by each axis. Keys 'axis_1' / 'axis_2' carry the sorted axis values.
    """
    das = _extract_common(scan, forcing, phyto_dim, zoo_dim, bin_geomeans,
                          bin_labels, sieburth_ranges, zoo_thresholds)
    a1 = scan[axis_name_1].values
    a2 = scan[axis_name_2].values

    out = {'axis_1': a1, 'axis_2': a2}
    for k, da in das.items():
        out[k] = da.transpose(axis_name_1, axis_name_2).values
    return out


# =============================================================================
# Single-point extraction (no scan dim) — for IVP comparisons against obs
# =============================================================================
def extract_single_point_metrics(out, forcing, tail=1000,
                                 phyto_dim='phyto', zoo_dim='zoo',
                                 bin_geomeans=CARIACO_PHYTO_BIN_GEOMEANS,
                                 bin_labels=DEFAULT_SIEBURTH_LABELS,
                                 sieburth_ranges=DEFAULT_SIEBURTH_RANGES,
                                 zoo_thresholds=DEFAULT_ZOO_THRESHOLDS):
    """Tail-mean diagnostic metrics from a single-point IVP output dataset.

    Same diagnostic variables as extract_scan_metrics_1d/2d, but for a single
    full-time-series IVP run (no scan dim). Tail-mean is taken over the last
    `tail` time steps. Returns a dict of scalar floats, keyed by variable:
    'mcs', 'P_pico', 'P_nano', 'P_micro', 'sumP', 'sumZ', 'Z_gt200',
    'Z_gt500', 'N', 'D', 'PP', 'Export'.
    """
    # Subset to just the variables we actually need before averaging — avoids
    # xarray trying to mean string-valued metadata variables.
    needed = ['Phytoplankton__biomass', 'Zooplankton__biomass',
              'Nutrient__value', 'Detritus__value',
              'Growth__uptake_value', 'DetritusSink__sinking_value']
    tw = out[needed].isel(time=slice(-tail, None))
    das = _extract_common(tw.mean('time'), forcing, phyto_dim, zoo_dim,
                          bin_geomeans, bin_labels, sieburth_ranges,
                          zoo_thresholds)
    res = {k: float(v.values) for k, v in das.items()}

    # Realised-stability scalars from the same tail window (nan-aware).
    cvP, hnP = _tail_cv(tw['Phytoplankton__biomass'].sum(dim=phyto_dim,
                                                          skipna=False).values)
    cvZ, hnZ = _tail_cv(tw['Zooplankton__biomass'].sum(dim=zoo_dim,
                                                       skipna=False).values)
    cvN, hnN = _tail_cv(tw['Nutrient__value'].values)
    res.update(cv_sumP=cvP, cv_sumZ=cvZ, cv_N=cvN,
               tail_has_nan=1.0 if (hnP or hnZ or hnN) else 0.0)
    return res


# =============================================================================
# Observation reference dict from monthly_df
# =============================================================================
def _safe_median(series):
    v = np.asarray(series).astype(float)
    v = v[np.isfinite(v)]
    return float(np.nanmedian(v)) if len(v) else np.nan


def build_obs_refs(monthly_df, bin_geomeans=CARIACO_PHYTO_BIN_GEOMEANS,
                   pico_col='pico_mmolN', nano_col='nano_mmolN',
                   micro_col='micro_mmolN',
                   z_gt200_col='zoo_gt200_mmolN', z_gt500_col='zoo_gt500_mmolN',
                   n_col='NO3_mmolN', pp_col='PP_mmolN_m3_d',
                   export_col='export_flux_corrected_mmolN'):
    """Build the per-target obs reference dict from monthly_df medians.

    mcs is computed from the per-month Pico/Nano/Micro bin fractions × bin
    geomeans (the same path the obs side uses), so model and obs mcs are
    methodologically aligned when the model side uses the Sieburth weights
    from compute_sieburth_weights.
    """
    P = monthly_df[[pico_col, nano_col, micro_col]].dropna().values
    sumP_obs = P.sum(axis=1)
    frac     = P / np.maximum(sumP_obs[:, None], 1e-30)
    mcs_obs  = 10.0 ** (frac @ np.log10(np.asarray(bin_geomeans)))

    return dict(
        mcs    = float(np.nanmedian(mcs_obs)),
        P_pico = float(np.nanmedian(P[:, 0])),
        P_nano = float(np.nanmedian(P[:, 1])),
        P_micro= float(np.nanmedian(P[:, 2])),
        sumP   = float(np.nanmedian(sumP_obs)),
        Z_gt200= _safe_median(monthly_df[z_gt200_col]),
        Z_gt500= _safe_median(monthly_df[z_gt500_col]),
        N      = _safe_median(monthly_df[n_col]),
        PP     = _safe_median(monthly_df[pp_col]),
        Export = _safe_median(monthly_df[export_col]),
    )


def build_obs_fraction_curve(monthly_df, fn_col='FN_mmolN_m2_d',
                             pico_col='pico_mmolN', nano_col='nano_mmolN',
                             micro_col='micro_mmolN', n_bins=6, min_per_bin=3):
    """Median Pico/Nano/Micro obs fraction as a function of F_N (quantile-binned).

    Returns a dict {'fn': (k,), 'pico'/'nano'/'micro': (k,)} of ascending
    bin-centre F_N and the median bin fractions, for *F_N-local* composition
    comparison — so the model curve is judged against the obs at the matching
    F_N rather than against a single pooled average (which mixes regimes and
    creates artifacts). Build it from the pooled ('all') monthly_df: low F_N
    bins are the relaxed/Pico end, high F_N bins the upwelling/Micro end.

    k ≤ n_bins (bins with < min_per_bin months are dropped). Returns None if
    there are too few usable months. Consumers (compute_flags) interpolate and
    clamp beyond the obs F_N coverage — flags past the last bin are
    extrapolations, to be read with care.
    """
    fn  = np.asarray(monthly_df[fn_col], dtype=float)
    P   = monthly_df[[pico_col, nano_col, micro_col]].to_numpy(dtype=float)
    tot = P.sum(axis=1)
    valid = np.isfinite(fn) & np.isfinite(tot) & (tot > 0)
    fn, frac = fn[valid], P[valid] / tot[valid, None]
    if fn.size < min_per_bin:
        return None
    edges = np.quantile(fn, np.linspace(0.0, 1.0, n_bins + 1))
    fnc, fr = [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (fn >= lo) & (fn <= hi) if i == n_bins - 1 else (fn >= lo) & (fn < hi)
        if int(m.sum()) >= min_per_bin:
            fnc.append(float(np.median(fn[m])))
            fr.append(np.median(frac[m], axis=0))
    if not fnc:
        return None
    order = np.argsort(fnc)
    fr = np.array(fr)[order]
    return dict(fn=np.array(fnc)[order], pico=fr[:, 0], nano=fr[:, 1], micro=fr[:, 2])


# =============================================================================
# Joint normalised L2 error (per scan point) — 1D or 2D
# =============================================================================
def joint_error(metrics, obs_refs, weights=None, targets=DEFAULT_TARGETS):
    """Normalised L2 error per scan point.

        err[p] = sqrt( Σ w_t · ((m_t[p] - o_t) / o_t)² / Σ w_t )

    summed over targets in `targets` where obs is finite and positive.
    `weights` is a dict mapping target name to weight (default 1.0 each).
    Returns a numpy array with the same shape as the per-variable metric
    arrays (1D or 2D).
    """
    if weights is None:
        weights = {t: 1.0 for t in targets}
    # Determine output shape from any one metric (they all share shape)
    shape = np.asarray(metrics[targets[0]]).shape
    num   = np.zeros(shape)
    wsum  = 0.0
    for t in targets:
        o = obs_refs.get(t, np.nan)
        if np.isfinite(o) and o > 0:
            w = weights.get(t, 1.0)
            num  = num + w * ((metrics[t] - o) / o) ** 2
            wsum = wsum + w
    return np.sqrt(num / wsum) if wsum > 0 else np.full(shape, np.nan)


# =============================================================================
# Flag layer (1D scans) — annotate and rank, NEVER filter
# =============================================================================
def compute_flags(metrics, obs_refs=None, thresholds=None, obs_frac_curve=None,
                  fn_eval_max=None, fn_micro_lo=3.5):
    """Admissibility + shape flags for a 1D scan. Annotates only — nothing is
    dropped, reordered, or hidden. A fit that matches obs but carries a flag
    still reaches the user; a clean fit is simply preferable.

    `obs_frac_curve` (from build_obs_fraction_curve) enables the F_N-local
    composition flags: the model Pico/Nano/Micro fractions are compared to the
    obs fractions interpolated to each scan-axis F_N, NOT to a pooled average
    (which mixes regimes). Requires the axis to be F_N; pass None to skip.

    `fn_eval_max` separates DISPLAY from GATING: the per-point flags below are
    always computed across the full sweep (so nothing is hidden), but the
    per-curve PASS/FAIL gates only judge the obs-covered window F_N ≤ fn_eval_max
    — so a run is never failed for behaviour in a thin, extrapolated high-F_N
    tail. None ⇒ the whole axis is the window.

    Returns a dict:
      per-point boolean arrays (full axis — for display):
        'collapse'     — tail NaN / sentinel / ΣP below floor
        'unstable'     — cv_sumP > cv_warn  (ran to completion, oscillating)
        'strong_cycle' — cv_sumP > cv_strong
        'n_runaway'    — N > n_ceiling
        'coexist_loss' — any Sieburth bin fraction < frac_floor
        'nano_over'    — model Nano fraction > obs Nano fraction(F_N)·(1+comp_tol)
                         (over-emphasises Nano vs obs at that F_N; needs obs_frac_curve)
        'micro_short'  — model Micro fraction < obs Micro fraction(F_N)·(1−comp_tol)
      per-curve gates (judged within F_N ≤ fn_eval_max):
        'mcs_descending'  — mcs not non-decreasing over the window
        'largeZ_thin'     — Z>200 below largeZ_obs_frac × obs across the window
        'nano_trough_ok'  — Flag B: Nano stays a trough (no nano_over in window);
                            None if obs_frac_curve unavailable
        'micro_ok_highFN' — Flag A: Micro is the dominant bin at the top of the
                            window (F_N ≥ fn_micro_lo); None if not evaluable
        'admissible_in_window' — no collapse/unstable/N-runaway/coexist in window
      'codes' : list[str] — compact per-point flag code (''=clean)
      'any'   : bool array — any per-point flag set at that point

    cv_sumP / tail_has_nan come from the avg_tail_stats postprocess; if the
    scan was run with plain avg_tail they are absent and the CV-based flags
    are simply not raised (reported as not-evaluated, not as 'clean').
    """
    th = {**FLAG_THRESHOLDS, **(thresholds or {})}
    axis = np.asarray(metrics['axis'])
    n = len(axis)
    axisf = np.asarray(axis, dtype=float)
    win = np.isfinite(axisf) if fn_eval_max is None else (np.isfinite(axisf) & (axisf <= fn_eval_max))

    def _arr(key):
        return np.asarray(metrics.get(key, np.full(n, np.nan)), dtype=float)

    cv    = _arr('cv_sumP')
    tnan  = _arr('tail_has_nan')
    sumP  = _arr('sumP')
    N     = _arr('N')
    pico  = _arr('P_pico'); nano = _arr('P_nano'); micro = _arr('P_micro')
    mcs   = _arr('mcs')

    have_cv = np.any(np.isfinite(cv)) or np.any(np.isfinite(tnan))

    # collapse: NaN tail / sentinel (tnan>=0.5 or non-finite cv where cv exists)
    # or a dead ΣP. Only assert the NaN side where CV info is present.
    collapse = (sumP < th['sumP_floor'])
    if have_cv:
        collapse = collapse | (tnan >= 0.5) | ~np.isfinite(cv)

    unstable     = np.isfinite(cv) & (cv > th['cv_warn'])   & ~collapse
    strong_cycle = np.isfinite(cv) & (cv > th['cv_strong']) & ~collapse
    n_runaway    = np.isfinite(N) & (N > th['n_ceiling'])

    bsum = np.where(sumP > 0, sumP, np.nan)
    fmin = np.minimum.reduce([pico, nano, micro]) / bsum
    coexist_loss = np.isfinite(fmin) & (fmin < th['frac_floor'])

    # F_N-local composition flags vs obs (need obs_frac_curve + an F_N axis).
    # Model Nano/Micro fractions compared to the obs fractions interpolated to
    # each F_N — so the bloom (high-F_N) end is judged against the upwelling obs,
    # not a pooled average.
    nano_over   = np.zeros(n, dtype=bool)
    micro_short = np.zeros(n, dtype=bool)
    if obs_frac_curve is not None and np.size(obs_frac_curve.get('fn', [])) >= 2:
        ofc   = obs_frac_curve
        denom = np.where(np.isfinite(bsum), bsum, np.nan)
        f_nano_mod  = nano  / denom
        f_micro_mod = micro / denom
        o_nano  = np.interp(axis, ofc['fn'], ofc['nano'])    # clamps beyond coverage
        o_micro = np.interp(axis, ofc['fn'], ofc['micro'])
        ctol = th['comp_tol']
        nano_over   = np.isfinite(f_nano_mod)  & (f_nano_mod  > o_nano  * (1 + ctol))
        micro_short = np.isfinite(f_micro_mod) & (f_micro_mod < o_micro * (1 - ctol))

    # ---- per-curve gates: judged only within the obs-covered window ----
    # mcs non-decreasing over the window (out-of-window tail never fails a run)
    fin = np.isfinite(mcs) & win
    mcs_descending = False
    if fin.sum() >= 2:
        m = mcs[fin]
        mcs_descending = bool(np.any(np.diff(m) <
                                     -th['mcs_desc_tol'] * np.maximum(m[:-1], 1e-9)))

    largeZ_thin = None
    o_z2 = obs_refs.get('Z_gt200', np.nan) if obs_refs else np.nan
    if np.isfinite(o_z2) and o_z2 > 0:
        z2 = _arr('Z_gt200'); z2w = z2[win]
        if np.any(np.isfinite(z2w)):
            largeZ_thin = bool(np.all(z2w[np.isfinite(z2w)] < th['largeZ_obs_frac'] * o_z2))

    # Flag B — Nano stays a trough: no nano_over inside the window
    nano_trough_ok = None
    if obs_frac_curve is not None and np.size(obs_frac_curve.get('fn', [])) >= 2:
        nano_trough_ok = not bool(np.any(nano_over & win))

    # Flag A — Micro dominant at the top of the window (F_N ≥ fn_micro_lo)
    micro_ok_highFN = None
    hiwin = win & (axisf >= fn_micro_lo)
    if np.any(hiwin):
        idx = np.where(hiwin)[0]
        top = idx[int(np.argmax(axisf[idx]))]
        micro_ok_highFN = bool(np.isfinite(micro[top]) and micro[top] > pico[top]
                               and micro[top] > nano[top])

    admissible_in_window = not bool(np.any(
        (collapse | unstable | n_runaway | coexist_loss) & win))

    codes = []
    for i in range(n):
        c = ''
        if collapse[i]:                 c += 'X'
        if strong_cycle[i]:             c += 'U!'
        elif unstable[i]:               c += 'U'
        if n_runaway[i]:                c += 'N'
        if coexist_loss[i]:             c += 'c'
        if nano_over[i]:                c += 'n'
        if micro_short[i]:              c += 'm'
        codes.append(c)

    any_pt = (collapse | unstable | strong_cycle | n_runaway | coexist_loss
              | nano_over | micro_short)

    return dict(collapse=collapse, unstable=unstable, strong_cycle=strong_cycle,
                n_runaway=n_runaway, coexist_loss=coexist_loss,
                nano_over=nano_over, micro_short=micro_short,
                mcs_descending=mcs_descending, largeZ_thin=largeZ_thin,
                nano_trough_ok=nano_trough_ok, micro_ok_highFN=micro_ok_highFN,
                admissible_in_window=admissible_in_window,
                codes=codes, any=any_pt, have_cv=have_cv)


def _flag_summary(flags, axis, axis_label='axis'):
    """One-line human-readable per-curve flag summary from compute_flags output.
    Code key: X=collapse, U/U!=oscillation(mild/strong), N=N-runaway,
    c=coexistence-loss, n=nano-over(vs obs), m=micro-short(vs obs),
    d=mcs-descending, z=large-Z-thin."""
    axis = np.asarray(axis)

    def _rng(mask, label):
        mask = np.asarray(mask)
        if mask.any():
            xs = axis[mask]
            return f"{label} {axis_label}∈[{xs.min():.2f},{xs.max():.2f}] (n={int(mask.sum())})"
        return None

    parts = [s for s in (
        _rng(flags['collapse'],     'collapse[X]'),
        _rng(flags['strong_cycle'], 'strong-cycle[U!]'),
        _rng(flags['unstable'] & ~flags['strong_cycle'], 'unstable[U]'),
        _rng(flags['n_runaway'],    'N-runaway[N]'),
        _rng(flags['coexist_loss'], 'coexist-loss[c]'),
        _rng(flags.get('nano_over', np.zeros(len(axis), bool)), 'nano-over[n]'),
        _rng(flags.get('micro_short', np.zeros(len(axis), bool)), 'micro-short[m]'),
    ) if s]
    if flags['mcs_descending']:
        parts.append('mcs-descending[d]')
    if flags['largeZ_thin']:
        parts.append('large-Z-thin[z]')
    if not flags.get('have_cv', True):
        parts.append('(CV not evaluated — run avg_tail_stats)')
    return 'FLAGS: ' + ('; '.join(parts) if parts else 'none')


# =============================================================================
# Formatters
# =============================================================================
def _fmt3(v):
    return f"{v:7.3f}" if np.isfinite(v) else f"{'-':>7}"


def _fmt4(v):
    return f"{v:7.4f}" if np.isfinite(v) else f"{'-':>7}"


def _fmt_mcs(v):
    return f"{v:6.2f}" if np.isfinite(v) else f"{'-':>6}"


# =============================================================================
# 1D table print
# =============================================================================
def print_scan_table_1d(metrics, obs_refs, header, axis_label='axis',
                        show_flags=True, obs_frac_curve=None):
    """Print a 1D scan table: one row per axis value, plus an obs row at the
    bottom. Columns: axis, mcs, Pico, Nano, Micro, ΣP, Z>200, Z>500, ΣZ,
    N, D, PP, Exp, CV (of ΣP), flag.

    The CV / flag columns and the trailing flag-summary line annotate each row
    with its realised-stability and shape caveats. Nothing is filtered — flags
    are information, not a cutoff. CV shows '-' if the scan was run with plain
    avg_tail (no cv_sumP); run avg_tail_stats to populate it.
    """
    axis = metrics['axis']
    cv = metrics.get('cv_sumP', None)
    flags = compute_flags(metrics, obs_refs, obs_frac_curve=obs_frac_curve) if show_flags else None
    print(f"\n=== {header} ===")
    print(f"{axis_label:>6} {'mcs':>6} {'Pico':>7} {'Nano':>7} {'Micro':>7} "
          f"{'ΣP':>7} {'Z>200':>7} {'Z>500':>7} {'ΣZ':>7} "
          f"{'N':>7} {'D':>7} {'PP':>7} {'Exp':>7} {'CV':>7} {'flag':>5}")
    for i in range(len(axis)):
        cv_s   = _fmt4(cv[i]) if cv is not None else f"{'-':>7}"
        code_s = flags['codes'][i] if flags is not None else ''
        print(f"{axis[i]:6.3f} {_fmt_mcs(metrics['mcs'][i])} "
              f"{_fmt3(metrics['P_pico'][i])} {_fmt3(metrics['P_nano'][i])} "
              f"{_fmt3(metrics['P_micro'][i])} {_fmt3(metrics['sumP'][i])} "
              f"{_fmt4(metrics['Z_gt200'][i])} {_fmt4(metrics['Z_gt500'][i])} "
              f"{_fmt3(metrics['sumZ'][i])} {_fmt3(metrics['N'][i])} "
              f"{_fmt3(metrics['D'][i])} {_fmt3(metrics['PP'][i])} "
              f"{_fmt3(metrics['Export'][i])} {cv_s} {code_s:>5}")
    print(f"{'obs':>6} {_fmt_mcs(obs_refs['mcs'])} "
          f"{_fmt3(obs_refs['P_pico'])} {_fmt3(obs_refs['P_nano'])} "
          f"{_fmt3(obs_refs['P_micro'])} {_fmt3(obs_refs['sumP'])} "
          f"{_fmt4(obs_refs['Z_gt200'])} {_fmt4(obs_refs['Z_gt500'])} "
          f"{'-':>7} {_fmt3(obs_refs['N'])} {'-':>7} "
          f"{_fmt3(obs_refs['PP'])} {_fmt3(obs_refs['Export'])} "
          f"{'-':>7} {'':>5}")
    if flags is not None:
        print(_flag_summary(flags, axis, axis_label))


# =============================================================================
# 2D table print — top-N best fits + a baseline row
# =============================================================================
def print_scan_table_2d(metrics, obs_refs, header, axis_labels,
                        n_top=5, baseline_axes=None,
                        weights_eq=None, weights_zw=None,
                        targets=DEFAULT_TARGETS):
    """Print a 2D scan summary: optional baseline row + top-N best fits
    under both equal-weighted and (optional) Z-weighted joint error.

    axis_labels : tuple (label_1, label_2) — column headers for the two
        scan axes.
    baseline_axes : tuple (val_1, val_2) — if given, print a row labelled
        'base' at the nearest grid point.
    weights_eq, weights_zw : optional dicts of per-target weights for two
        separate rankings. If weights_zw is None, only the equal-weighted
        ranking is printed.
    """
    a1 = metrics['axis_1']
    a2 = metrics['axis_2']
    lab1, lab2 = axis_labels

    err_eq = joint_error(metrics, obs_refs, weights=weights_eq, targets=targets)
    if weights_zw is not None:
        err_zw = joint_error(metrics, obs_refs, weights=weights_zw,
                             targets=targets)
    else:
        err_zw = None

    print(f"\n=== {header} ===")
    print(f"{'rank':>6} {lab1:>6} {lab2:>7}  {'mcs':>6} "
          f"{'Pico':>7} {'Nano':>7} {'Micro':>7} "
          f"{'Z>200':>7} {'Z>500':>7} {'err_eq':>7}"
          + (f" {'err_zw':>7}" if err_zw is not None else ""))

    def _row(i, j, label):
        s = (f"{label:>6} {a1[i]:6.2f} {a2[j]:7.2f}  "
             f"{_fmt_mcs(metrics['mcs'][i, j])} "
             f"{_fmt3(metrics['P_pico'][i, j])} {_fmt3(metrics['P_nano'][i, j])} "
             f"{_fmt3(metrics['P_micro'][i, j])} "
             f"{_fmt4(metrics['Z_gt200'][i, j])} "
             f"{_fmt4(metrics['Z_gt500'][i, j])} "
             f"{_fmt3(err_eq[i, j])}")
        if err_zw is not None:
            s += f" {_fmt3(err_zw[i, j])}"
        return s

    if baseline_axes is not None:
        i_b = int(np.argmin(np.abs(a1 - baseline_axes[0])))
        j_b = int(np.argmin(np.abs(a2 - baseline_axes[1])))
        print(_row(i_b, j_b, 'base'))

    # obs row (a placeholder — axis cols are '-')
    print(f"{'obs':>6} {'-':>6} {'-':>7}  {_fmt_mcs(obs_refs['mcs'])} "
          f"{_fmt3(obs_refs['P_pico'])} {_fmt3(obs_refs['P_nano'])} "
          f"{_fmt3(obs_refs['P_micro'])} "
          f"{_fmt4(obs_refs['Z_gt200'])} {_fmt4(obs_refs['Z_gt500'])}  "
          + (f"{'-':>6}" + (f"  {'-':>6}" if err_zw is not None else "")))

    for tag, err in [('Top-{n} equal-weighted err_eq'.format(n=n_top), err_eq)] + \
                    ([('Top-{n} Z-weighted err_zw'.format(n=n_top), err_zw)]
                     if err_zw is not None else []):
        print(f"\n{tag}:")
        flat_idx = np.argsort(err.ravel())[:n_top]
        for rank, k in enumerate(flat_idx, start=1):
            ii, jj = np.unravel_index(k, err.shape)
            print(_row(ii, jj, f"#{rank}"))


# =============================================================================
# Single-point comparison table (one or more model runs vs obs)
# =============================================================================
def print_single_point_comparison(metrics_dict, obs_refs, header,
                                  label_col_width=10):
    """Print a comparison table with one or more labelled single-point runs
    plus an obs reference row, in the same column layout as
    print_scan_table_1d.

    Parameters
    ----------
    metrics_dict : dict {label: metrics_scalar_dict}
        Each value is a dict of scalars as returned by
        extract_single_point_metrics. Insertion order is the row order.
    obs_refs : dict
        Per-target obs medians from build_obs_refs.
    header : str
        Printed above the table (regime / config description).
    label_col_width : int, optional
        Width of the leftmost label column. Default 10.
    """
    print(f"\n=== {header} ===")
    print(f"{'config':>{label_col_width}} {'mcs':>6} {'Pico':>7} {'Nano':>7} "
          f"{'Micro':>7} {'ΣP':>7} {'Z>200':>7} {'Z>500':>7} {'ΣZ':>7} "
          f"{'N':>7} {'D':>7} {'PP':>7} {'Exp':>7}")

    def _row(label, m):
        return (f"{label:>{label_col_width}} "
                f"{_fmt_mcs(m.get('mcs', np.nan))} "
                f"{_fmt3(m.get('P_pico', np.nan))} {_fmt3(m.get('P_nano', np.nan))} "
                f"{_fmt3(m.get('P_micro', np.nan))} {_fmt3(m.get('sumP', np.nan))} "
                f"{_fmt4(m.get('Z_gt200', np.nan))} {_fmt4(m.get('Z_gt500', np.nan))} "
                f"{_fmt3(m.get('sumZ', np.nan))} {_fmt3(m.get('N', np.nan))} "
                f"{_fmt3(m.get('D', np.nan))} {_fmt3(m.get('PP', np.nan))} "
                f"{_fmt3(m.get('Export', np.nan))}")

    for label, m in metrics_dict.items():
        print(_row(label, m))
    # Obs row — same column layout; sumZ and D have no obs.
    obs_metrics = dict(
        mcs=obs_refs.get('mcs', np.nan),
        P_pico=obs_refs.get('P_pico', np.nan),
        P_nano=obs_refs.get('P_nano', np.nan),
        P_micro=obs_refs.get('P_micro', np.nan),
        sumP=obs_refs.get('sumP', np.nan),
        Z_gt200=obs_refs.get('Z_gt200', np.nan),
        Z_gt500=obs_refs.get('Z_gt500', np.nan),
        sumZ=np.nan,
        N=obs_refs.get('N', np.nan),
        D=np.nan,
        PP=obs_refs.get('PP', np.nan),
        Export=obs_refs.get('Export', np.nan),
    )
    print(_row('obs', obs_metrics))


# =============================================================================
# 1D 6-panel diagnostic plot
# =============================================================================
def plot_scan_panels_1d(metrics, obs_refs, header, axis_label='axis',
                        figsize=(20, 9), show_flags=True, obs_frac_curve=None):
    """8-panel diagnostic plot for a 1D scan (2×4). Linear axes (CV log-y);
    dashed lines = obs medians where available. The 7th panel is CV(ΣP) with
    the WARN/STRONG stability guides; the 8th is the per-curve flag summary.
    Flagged points are ringed on the mcs panel. Annotates, never filters."""
    axis = metrics['axis']
    flags = compute_flags(metrics, obs_refs, obs_frac_curve=obs_frac_curve) if show_flags else None
    fig, axes = plt.subplots(2, 4, figsize=figsize)
    fig.suptitle(header)

    # Panel 1: mcs
    ax = axes[0, 0]
    ax.plot(axis, metrics['mcs'], '-o', color='C0')
    if np.isfinite(obs_refs.get('mcs', np.nan)):
        ax.axhline(obs_refs['mcs'], color='k', ls='--', lw=1, label='obs')
        ax.legend(fontsize=9)
    if flags is not None and np.any(flags['any']):
        m = flags['any']
        ax.plot(np.asarray(axis)[m], np.asarray(metrics['mcs'])[m], 'o',
                mfc='none', mec='red', mew=1.5, ms=12, label='flagged')
    ax.set_xlabel(axis_label); ax.set_ylabel('mcs [µm]')
    ax.set_title('mean cell size')

    # Panel 2: phyto bands
    ax = axes[0, 1]
    for lab, key, color in [('Pico', 'P_pico', 'C0'),
                            ('Nano', 'P_nano', 'C1'),
                            ('Micro', 'P_micro', 'C2')]:
        ax.plot(axis, metrics[key], '-o', color=color, label=lab)
        if np.isfinite(obs_refs.get(key, np.nan)):
            ax.axhline(obs_refs[key], color=color, ls='--', lw=1)
    ax.set_xlabel(axis_label); ax.set_ylabel('biomass [mmol N m⁻³]')
    ax.set_title('phyto bands (dashed = obs)'); ax.legend(fontsize=9)

    # Panel 3: large-Z bands
    ax = axes[0, 2]
    for lab, key, color in [('Z>200', 'Z_gt200', 'C3'),
                            ('Z>500', 'Z_gt500', 'C4')]:
        ax.plot(axis, metrics[key], '-o', color=color, label=lab)
        if np.isfinite(obs_refs.get(key, np.nan)):
            ax.axhline(obs_refs[key], color=color, ls='--', lw=1)
    ax.set_xlabel(axis_label); ax.set_ylabel('biomass [mmol N m⁻³]')
    ax.set_title('large-Z bands (depth-mismatch caveat)'); ax.legend(fontsize=9)

    # Panel 4: totals + N + D
    ax = axes[1, 0]
    ax.plot(axis, metrics['sumP'], '-o', color='C0', label='ΣP')
    ax.plot(axis, metrics['sumZ'], '-o', color='C2', label='ΣZ')
    ax.plot(axis, metrics['N'],    '-o', color='C1', label='N')
    ax.plot(axis, metrics['D'],    '-o', color='C5', label='D')
    if np.isfinite(obs_refs.get('sumP', np.nan)):
        ax.axhline(obs_refs['sumP'], color='C0', ls='--', lw=1)
    if np.isfinite(obs_refs.get('N', np.nan)):
        ax.axhline(obs_refs['N'], color='C1', ls='--', lw=1)
    ax.set_xlabel(axis_label); ax.set_ylabel('mmol N m⁻³')
    ax.set_title('totals + N + D (dashed = obs where available)')
    ax.legend(fontsize=9)

    # Panel 5: PP
    ax = axes[1, 1]
    ax.plot(axis, metrics['PP'], '-o', color='C8')
    if np.isfinite(obs_refs.get('PP', np.nan)):
        ax.axhline(obs_refs['PP'], color='k', ls='--', lw=1, label='obs')
        ax.legend(fontsize=9)
    ax.set_xlabel(axis_label); ax.set_ylabel('PP [mmol N m⁻³ d⁻¹]')
    ax.set_title('primary productivity')

    # Panel 6: Export
    ax = axes[1, 2]
    ax.plot(axis, metrics['Export'], '-o', color='C6')
    if np.isfinite(obs_refs.get('Export', np.nan)):
        ax.axhline(obs_refs['Export'], color='k', ls='--', lw=1, label='obs')
        ax.legend(fontsize=9)
    ax.set_xlabel(axis_label); ax.set_ylabel('Export [mmol N m⁻² d⁻¹]')
    ax.set_title('detritus sinking export')

    # Panel 7: CV(ΣP) — realised stability (log-y, WARN/STRONG guides)
    ax = axes[0, 3]
    cv = metrics.get('cv_sumP', None)
    if cv is not None and np.any(np.isfinite(cv)):
        cvp = np.where(np.isfinite(np.asarray(cv, float)) &
                       (np.asarray(cv, float) > 0), cv, np.nan)
        ax.plot(axis, cvp, '-o', color='C5')
        ax.axhline(FLAG_THRESHOLDS['cv_warn'], color='orange', ls='--', lw=1,
                   label=f"warn {FLAG_THRESHOLDS['cv_warn']}")
        ax.axhline(FLAG_THRESHOLDS['cv_strong'], color='red', ls='--', lw=1,
                   label=f"strong {FLAG_THRESHOLDS['cv_strong']}")
        ax.set_yscale('log'); ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, 'CV unavailable\n(run avg_tail_stats)', ha='center',
                va='center', transform=ax.transAxes, fontsize=10)
    ax.set_xlabel(axis_label); ax.set_ylabel('CV(ΣP)')
    ax.set_title('realised stability')

    # Panel 8: per-curve flag summary
    ax = axes[1, 3]; ax.axis('off'); ax.set_title('flags')
    if flags is not None:
        txt = _flag_summary(flags, axis, axis_label).replace('; ', ';\n')
        ax.text(0.02, 0.98, txt, ha='left', va='top', transform=ax.transAxes,
                fontsize=9, family='monospace')

    plt.tight_layout()
    return fig


# =============================================================================
# 2D heatmap grid (size-spectrum targets)
# =============================================================================
def plot_2d_heatmaps(metrics, obs_refs, header, axis_labels,
                     weights_eq=None, weights_zw=None,
                     targets=DEFAULT_TARGETS, cmap='viridis',
                     figsize=(16, 9)):
    """6-panel heatmap grid for a 2D scan: mcs, Pico, Nano, Micro, Z>200, Z>500.

    Each panel is a pcolormesh of the model value on the (axis_1, axis_2) grid
    with a black contour at the obs value (only if the obs is inside the
    grid's model-value range). Markers indicate the joint-error minima:
    white ★ = equal-weighted minimum, red ◆ = Z-weighted minimum (only if
    weights_zw is given).
    """
    a1 = metrics['axis_1']
    a2 = metrics['axis_2']
    lab1, lab2 = axis_labels

    err_eq = joint_error(metrics, obs_refs, weights=weights_eq, targets=targets)
    if weights_zw is not None:
        err_zw = joint_error(metrics, obs_refs, weights=weights_zw,
                             targets=targets)
        i_zw, j_zw = np.unravel_index(np.nanargmin(err_zw), err_zw.shape)
    else:
        i_zw = j_zw = None

    i_eq, j_eq = np.unravel_index(np.nanargmin(err_eq), err_eq.shape)

    panels = [
        ('mcs [µm]',          metrics['mcs'],     obs_refs.get('mcs', np.nan)),
        ('Pico [mmolN m⁻³]',  metrics['P_pico'],  obs_refs.get('P_pico', np.nan)),
        ('Nano [mmolN m⁻³]',  metrics['P_nano'],  obs_refs.get('P_nano', np.nan)),
        ('Micro [mmolN m⁻³]', metrics['P_micro'], obs_refs.get('P_micro', np.nan)),
        ('Z>200 [mmolN m⁻³]', metrics['Z_gt200'], obs_refs.get('Z_gt200', np.nan)),
        ('Z>500 [mmolN m⁻³]', metrics['Z_gt500'], obs_refs.get('Z_gt500', np.nan)),
    ]

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    suptitle = f"{header}   ★ = err_eq min"
    if i_zw is not None:
        suptitle += ", ◆ = err_zw min"
    fig.suptitle(suptitle)

    for ax, (title, data, ov) in zip(axes.flat, panels):
        # data shape is (n_a1, n_a2); pcolormesh wants C as (n_y, n_x)
        # We put a1 on x-axis and a2 on y-axis → C = data.T (n_a2, n_a1)
        pc = ax.pcolormesh(a1, a2, data.T, shading='auto', cmap=cmap)
        fig.colorbar(pc, ax=ax)
        if np.isfinite(ov) and (data.min() <= ov <= data.max()):
            cs = ax.contour(a1, a2, data.T, levels=[ov],
                            colors='k', linewidths=1.8)
            ax.clabel(cs, fmt={ov: 'obs'}, fontsize=9, inline=True)
        ax.plot(a1[i_eq], a2[j_eq], '*', color='white', mec='black', ms=14)
        if i_zw is not None:
            ax.plot(a1[i_zw], a2[j_zw], 'D', color='red', mec='black', ms=9)
        ax.set_xlabel(lab1); ax.set_ylabel(lab2); ax.set_title(title)

    plt.tight_layout()
    return fig
