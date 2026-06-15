"""
parscan_utils_extended.py — extends parscan_utils.py with the per-target
diagnostic patterns we developed for the MS3 baseline (2026-06-14).

Reuses from parscan_utils:
- get_log_bin_edges, get_fraction_in_range
- CARIACO_PHYTO_BIN_GEOMEANS
- run_single_point (re-exported for convenience)

Adds (this file):
- compute_sieburth_weights(phyto_esd) -> (3, n_phyto) overlap-weight matrix
- extract_scan_metrics_1d(scan, axis_name, forcing) -> dict of per-variable arrays
- extract_scan_metrics_2d(scan, axis_name_1, axis_name_2, forcing) -> dict
- build_obs_refs(monthly_df) -> dict of per-target obs medians
- joint_error(metrics, obs_refs, weights=None, targets=...) -> per-point L2 error
- print_scan_table_1d(metrics, obs_refs, header, axis_label)
- print_scan_table_2d(metrics, obs_refs, header, axis_labels, n_top=5,
                      weights_eq=..., weights_zw=...)
- plot_scan_panels_1d(metrics, obs_refs, header, axis_label)
- plot_2d_heatmaps(metrics, obs_refs, header, axis_labels)

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
# Sieburth bin conventions (defaults match parscan_utils / cariaco_obs)
# =============================================================================
DEFAULT_SIEBURTH_RANGES = [(0.0, 2.0), (2.0, 20.0), (20.0, np.inf)]  # Pico, Nano, Micro
DEFAULT_SIEBURTH_LABELS = ['Pico', 'Nano', 'Micro']
DEFAULT_ZOO_THRESHOLDS = (200.0, 500.0)   # cumulative >threshold bands

# Size-spectrum targets we score against obs by default
DEFAULT_TARGETS = ['mcs', 'P_pico', 'P_nano', 'P_micro',
                   'Z_gt200', 'Z_gt500']


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

    return dict(
        mcs=mcs_da, P_pico=P_pico_da, P_nano=P_nano_da, P_micro=P_micro_da,
        sumP=sumP_da, sumZ=sumZ_da,
        N=N_da, D=D_da, PP=PP_da, Export=Export_da,
        **Z_band_das,
    )


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
def print_scan_table_1d(metrics, obs_refs, header, axis_label='axis'):
    """Print a 1D scan table: one row per axis value, plus an obs row at the
    bottom. Columns: axis, mcs, Pico, Nano, Micro, ΣP, Z>200, Z>500, ΣZ,
    N, D, PP, Exp.
    """
    axis = metrics['axis']
    print(f"\n=== {header} ===")
    print(f"{axis_label:>6} {'mcs':>6} {'Pico':>7} {'Nano':>7} {'Micro':>7} "
          f"{'ΣP':>7} {'Z>200':>7} {'Z>500':>7} {'ΣZ':>7} "
          f"{'N':>7} {'D':>7} {'PP':>7} {'Exp':>7}")
    for i in range(len(axis)):
        print(f"{axis[i]:6.3f} {_fmt_mcs(metrics['mcs'][i])} "
              f"{_fmt3(metrics['P_pico'][i])} {_fmt3(metrics['P_nano'][i])} "
              f"{_fmt3(metrics['P_micro'][i])} {_fmt3(metrics['sumP'][i])} "
              f"{_fmt4(metrics['Z_gt200'][i])} {_fmt4(metrics['Z_gt500'][i])} "
              f"{_fmt3(metrics['sumZ'][i])} {_fmt3(metrics['N'][i])} "
              f"{_fmt3(metrics['D'][i])} {_fmt3(metrics['PP'][i])} "
              f"{_fmt3(metrics['Export'][i])}")
    print(f"{'obs':>6} {_fmt_mcs(obs_refs['mcs'])} "
          f"{_fmt3(obs_refs['P_pico'])} {_fmt3(obs_refs['P_nano'])} "
          f"{_fmt3(obs_refs['P_micro'])} {_fmt3(obs_refs['sumP'])} "
          f"{_fmt4(obs_refs['Z_gt200'])} {_fmt4(obs_refs['Z_gt500'])} "
          f"{'-':>7} {_fmt3(obs_refs['N'])} {'-':>7} "
          f"{_fmt3(obs_refs['PP'])} {_fmt3(obs_refs['Export'])}")


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
# 1D 6-panel diagnostic plot
# =============================================================================
def plot_scan_panels_1d(metrics, obs_refs, header, axis_label='axis',
                        figsize=(16, 9)):
    """6-panel diagnostic plot for a 1D scan. Linear axes; dashed lines = obs
    medians where available."""
    axis = metrics['axis']
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(header)

    # Panel 1: mcs
    ax = axes[0, 0]
    ax.plot(axis, metrics['mcs'], '-o', color='C0')
    if np.isfinite(obs_refs.get('mcs', np.nan)):
        ax.axhline(obs_refs['mcs'], color='k', ls='--', lw=1, label='obs')
        ax.legend(fontsize=9)
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
