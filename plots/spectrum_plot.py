"""
spectrum_plot.py
================

Normalised biomass spectrum plotting for MS3 Cariaco — model steady-state
output vs Cariaco observations, with optional octave-binned model community
spectrum for Sheldon-flat comparison.

Single entry point:
    plot_normalised_biomass_spectrum(...)

Layer combinations (via two booleans):
    show_zoo=True,  show_octave=True   →  full plot (default)
    show_zoo=True,  show_octave=False  →  per-class + 5-bin overlay only
    show_zoo=False, show_octave=False  →  phyto-only (3 bins)
    show_zoo=False, show_octave=True   →  phyto-only with octave (allowed)

Imports cariaco_obs.load_cariaco_targets and
parscan_utils.{get_log_bin_edges, get_fraction_in_range}, so this file
must live alongside those modules on the import path.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# This file lives in <repo>/plots/. Its dependencies (cariaco_obs, parscan_utils)
# live in the sibling folder <repo>/parameter_scan/. Add that to sys.path so the
# imports below resolve regardless of where Python was launched from.
_PARSCAN_DIR = Path(__file__).resolve().parent.parent / 'parameter_scan'
if str(_PARSCAN_DIR) not in sys.path:
    sys.path.insert(0, str(_PARSCAN_DIR))

from cariaco_obs import load_cariaco_targets
from parscan_utils import get_log_bin_edges, get_fraction_in_range


__all__ = ['plot_normalised_biomass_spectrum']


# Defaults (overridable via function args)
DEFAULT_EXTINCT_THRESHOLD = 1e-7   # mmol N m^-3 per class
DEFAULT_OCTAVE_THRESHOLD  = 1e-6   # mmol N m^-3 per bin
CAP_LOG = 0.04                     # whisker cap half-width in log-decades


# =============================================================================
# Internal helpers
# =============================================================================

def _make_integrators(ss_phyto_arr, ss_zoo_arr, p_edges, z_edges):
    """Closures: fraction-weighted integration of model classes within a range."""
    n_p, n_z = len(ss_phyto_arr), len(ss_zoo_arr)

    def phyto_in(smin, smax):
        return sum(ss_phyto_arr[i] * get_fraction_in_range(
            p_edges[i], p_edges[i+1], smin, smax) for i in range(n_p))

    def zoo_in(smin, smax):
        return sum(ss_zoo_arr[i] * get_fraction_in_range(
            z_edges[i], z_edges[i+1], smin, smax) for i in range(n_z))

    return phyto_in, zoo_in


def _build_plot_bins(phyto_in, zoo_in, monthly_df, bin_defs, obs_vec,
                     phyto_lower, phyto_upper, zoo_upper, show_zoo):
    """Construct 3-bin (phyto only) or 5-bin (P+Z) plot definitions.

    Obs zoo bins are reconstructed via per-month net-tow subtraction
    (>200 minus >500); model zoo bins are direct integration on the same
    ranges (no subtraction needed — model has full grid)."""
    by_label = {b['label']: i for i, b in enumerate(bin_defs)}

    def obs(lbl):
        return obs_vec[by_label[lbl]]

    def monthly(lbl):
        return monthly_df[bin_defs[by_label[lbl]]['column']]

    bins = [
        {'label': 'Pico (0.5–2 µm)', 'type': 'phyto',
         'smin': phyto_lower, 'smax': 2.0,
         'obs':     obs('Pico (<2 µm)'),
         'monthly': monthly('Pico (<2 µm)'),
         'model':   phyto_in(phyto_lower, 2.0)},
        {'label': 'Nano (2–20 µm)', 'type': 'phyto',
         'smin': 2.0, 'smax': 20.0,
         'obs':     obs('Nano (2-20 µm)'),
         'monthly': monthly('Nano (2-20 µm)'),
         'model':   phyto_in(2.0, 20.0)},
        {'label': 'Micro (20–200 µm)', 'type': 'phyto',
         'smin': 20.0, 'smax': phyto_upper,
         'obs':     obs('Micro (>20 µm)'),
         'monthly': monthly('Micro (>20 µm)'),
         'model':   phyto_in(20.0, phyto_upper)},
    ]

    if show_zoo:
        # Per-month subtraction → consistent central + min/max
        zoo_200_500 = (
            monthly('Zoo >200 µm') - monthly('Zoo >500 µm')
        ).clip(lower=0.0)
        bins.extend([
            {'label': 'Zoo (200–500 µm]', 'type': 'zoo',
             'smin': 200.0, 'smax': 500.0,
             'obs':     max(zoo_200_500.dropna().mean(), 0.0),
             'monthly': zoo_200_500,
             'model':   zoo_in(200.0, 500.0)},
            {'label': 'Zoo (500–2000 µm]', 'type': 'zoo',
             'smin': 500.0, 'smax': zoo_upper,
             'obs':     obs('Zoo >500 µm'),
             'monthly': monthly('Zoo >500 µm'),
             'model':   zoo_in(500.0, zoo_upper)},
        ])
    return bins


def _segments_and_whiskers(plot_bins):
    """Convert plot-bin defs into segment + whisker dicts (in density units)."""
    obs_segments, model_segments, obs_whiskers = [], [], []
    for b in plot_bins:
        smin, smax = b['smin'], b['smax']
        dlog = np.log10(smax) - np.log10(smin)
        geo  = np.sqrt(smin * smax)

        obs_segments.append({
            'size_min': smin, 'size_max': smax,
            'density':  b['obs'] / dlog, 'type': b['type']})
        model_segments.append({
            'size_min': smin, 'size_max': smax,
            'density':  b['model'] / dlog, 'type': b['type']})

        monthly_vals = b['monthly'].dropna()
        monthly_vals = monthly_vals[monthly_vals > 0].to_numpy()
        if monthly_vals.size:
            obs_whiskers.append({
                'x':     geo,
                'y_min': monthly_vals.min() / dlog,
                'y_max': monthly_vals.max() / dlog,
                'type':  b['type']})
    return obs_segments, model_segments, obs_whiskers


def _octave_community(phyto_in, zoo_in, edges, threshold):
    """Aggregate model P+Z onto octave bins; mask sub-threshold (empty) bins."""
    smin = edges[:-1]
    smax = edges[1:]
    geo  = np.sqrt(smin * smax)
    dlog = np.log10(smax) - np.log10(smin)

    p = np.array([phyto_in(s, e) for s, e in zip(smin, smax)])
    z = np.array([zoo_in(s, e)   for s, e in zip(smin, smax)])
    comm = p + z
    density = np.where(comm > threshold, comm / dlog, np.nan)
    masked  = np.where(np.isnan(density))[0]
    return geo, density, masked


# =============================================================================
# Main entry point
# =============================================================================

def plot_normalised_biomass_spectrum(
    ss_phyto, ss_zoo, phyto_esd, zoo_esd,
    *,
    regime='all',
    show_zoo=True,
    show_octave=True,
    extinct_threshold=DEFAULT_EXTINCT_THRESHOLD,
    octave_threshold=DEFAULT_OCTAVE_THRESHOLD,
    octave_edges=None,
    ax=None,
    figsize=(10, 6),
    title=None,
):
    """
    Plot the normalised biomass spectrum for a model steady-state output.

    Parameters
    ----------
    ss_phyto, ss_zoo : array-like
        Per-class steady-state biomass (mmol N m^-3).
    phyto_esd, zoo_esd : array-like
        Per-class size centres (µm ESD).
    regime : str, default 'all'
        Cariaco regime — 'all' / 'upwelling' / 'relaxed' / 'strong' /
        'moderate' / 'weak' (passed to load_cariaco_targets).
    show_zoo : bool, default True
        If False, hide all zoo elements (per-class points, zoo bins, zoo
        whiskers). Plot becomes phyto-only.
    show_octave : bool, default True
        If True, overlay the octave-binned model community spectrum
        (P+Z, or phyto-only when show_zoo=False).
    extinct_threshold : float, default 1e-7
        Per-class biomass threshold for masking the per-class scatter
        (visual cleanliness only — does not affect bin aggregation).
    octave_threshold : float, default 1e-6
        Per-bin biomass threshold below which an octave bin is treated
        as empty (excluded from line, marked with × at axis bottom).
    octave_edges : array-like or None
        Custom octave bin edges in µm. None → factor-2 octaves from
        0.5 to 2048 µm (12 bins).
    ax : matplotlib Axes or None
        If None, a new figure is created.
    figsize : tuple
        Figure size when creating a new figure.
    title : str or None
        Custom title; if None an auto-generated one is used.

    Returns
    -------
    ax : matplotlib Axes
    """
    ss_phyto_arr  = np.asarray(ss_phyto)
    ss_zoo_arr    = np.asarray(ss_zoo)
    phyto_esd_arr = np.asarray(phyto_esd)
    zoo_esd_arr   = np.asarray(zoo_esd)

    # --- Log-bin geometry ---
    p_edges = get_log_bin_edges(phyto_esd_arr)
    z_edges = get_log_bin_edges(zoo_esd_arr)
    p_dlog  = np.diff(np.log10(p_edges))
    z_dlog  = np.diff(np.log10(z_edges))

    # --- Per-class NBS (extinct-class masking is visual only) ---
    phyto_clean = np.where(ss_phyto_arr < extinct_threshold, np.nan, ss_phyto_arr)
    zoo_clean   = np.where(ss_zoo_arr   < extinct_threshold, np.nan, ss_zoo_arr)
    phyto_nbs = phyto_clean / p_dlog
    zoo_nbs   = zoo_clean   / z_dlog

    # --- Cariaco obs ---
    obs_vec, _, bin_defs, monthly_df, _ = load_cariaco_targets(regime=regime)

    # --- Caps from model grid ---
    PHYTO_LOWER = float(p_edges[0])    # ~0.38 µm — actual model lower edge
    PHYTO_UPPER = float(p_edges[-1])   # ~264 µm — actual model upper edge
    ZOO_UPPER   = float(z_edges[-1])   # ~2640 µm — actual model upper edge

    # --- Integrators + plot bins ---
    phyto_in, zoo_in = _make_integrators(ss_phyto_arr, ss_zoo_arr, p_edges, z_edges)
    plot_bins = _build_plot_bins(
        phyto_in, zoo_in, monthly_df, bin_defs, obs_vec,
        PHYTO_LOWER, PHYTO_UPPER, ZOO_UPPER, show_zoo,
    )
    obs_segments, model_segments, obs_whiskers = _segments_and_whiskers(plot_bins)

    # --- Octave community (optional) ---
    octave_geo = octave_density = masked_octave = None
    if show_octave:
        if octave_edges is None:
            octave_edges = np.array([0.5 * 2**i for i in range(13)])  # 0.5–2048 µm
        else:
            octave_edges = np.asarray(octave_edges)
        octave_geo, octave_density, masked_octave = _octave_community(
            phyto_in, zoo_in, octave_edges, octave_threshold,
        )

    # === Plot ===
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    # Per-class points
    ax.scatter(phyto_esd_arr, phyto_nbs, marker='o', s=70,
               color='tab:green', edgecolor='black', linewidth=0.5,
               zorder=5, label='Model phyto (per class)')
    if show_zoo:
        ax.scatter(zoo_esd_arr, zoo_nbs, marker='s', s=70,
                   color='tab:red', edgecolor='black', linewidth=0.5,
                   zorder=5, label='Model zoo (per class)')

    # Cariaco bin central tendency (faded thick bar)
    for seg in obs_segments:
        c = 'tab:green' if seg['type'] == 'phyto' else 'tab:red'
        ax.hlines(seg['density'], seg['size_min'], seg['size_max'],
                  colors=c, linewidth=4, alpha=0.35, zorder=3)

    # Cariaco monthly min–max whiskers (vertical line + cap)
    for w in obs_whiskers:
        c = 'tab:green' if w['type'] == 'phyto' else 'tab:red'
        ax.vlines(w['x'], w['y_min'], w['y_max'],
                  colors=c, linewidth=1.2, alpha=0.8, zorder=3.5)
        x_lo, x_hi = w['x'] * 10**(-CAP_LOG), w['x'] * 10**(+CAP_LOG)
        ax.hlines([w['y_min'], w['y_max']], x_lo, x_hi,
                  colors=c, linewidth=1.2, alpha=0.8, zorder=3.5)

    # Bin-aggregated model overlay (dashed)
    for seg in model_segments:
        c = 'tab:green' if seg['type'] == 'phyto' else 'tab:red'
        ax.hlines(seg['density'], seg['size_min'], seg['size_max'],
                  colors=c, linestyles='dashed', linewidth=2.0,
                  alpha=1.0, zorder=4)

    # Octave-binned community
    if show_octave:
        comm_label = ('Model community (octave bins, P+Z)' if show_zoo
                      else 'Model community (octave bins, phyto)')
        ax.plot(octave_geo, octave_density,
                marker='D', markersize=7, linestyle='-', linewidth=1.2,
                color='navy', markerfacecolor='navy', markeredgecolor='black',
                zorder=4.5, label=comm_label)
        if len(masked_octave) > 0:
            ax.scatter(octave_geo[masked_octave],
                       np.full(len(masked_octave), 0.04),
                       transform=ax.get_xaxis_transform(),
                       marker='x', s=80, color='navy', alpha=0.7, zorder=4.5,
                       label='Octave bin below threshold (empty)')

    # Sheldon-flat reference (anchored at phyto obs centroid)
    phyto_obs = [s for s in obs_segments if s['type'] == 'phyto']
    y_anchor  = np.exp(np.mean(np.log([s['density'] for s in phyto_obs])))

    ax.axhline(y_anchor, color='gray', linestyle='--', linewidth=1,
               alpha=0.7, label='Sheldon-flat (community ref.)')

    # Empirical phyto bin slope — polyfit through the 3 phyto bin densities
    # (in /Δlog₁₀ normalization, matching the y-axis). Drawn over phyto range.
    phyto_model = [s for s in model_segments if s['type'] == 'phyto']
    phyto_x = np.array([np.sqrt(s['size_min'] * s['size_max'])
                        for s in phyto_model])
    phyto_y = np.array([s['density'] for s in phyto_model])
    valid = phyto_y > 0
    if valid.sum() >= 2:
        slope_p, intercept_p = np.polyfit(np.log10(phyto_x[valid]),
                                          np.log10(phyto_y[valid]), 1)
        x_fit = np.array([phyto_esd_arr[0], phyto_esd_arr[-1]])
        y_fit = 10 ** (intercept_p + slope_p * np.log10(x_fit))
        ax.plot(x_fit, y_fit, color='darkgreen', linestyle='--', linewidth=1.,
                alpha=0.6, zorder=4.5,
                label=f'Phyto bin slope ({slope_p:+.2f})')

    # Pico / Nano / Micro size-class guides
    for x_guide in (2.0, 20.0):
        ax.axvline(x_guide, color='gray', linestyle=':', linewidth=0.5, alpha=0.4)

    # Legend (auto handles + manual handles for the looped overlays)
    extra_handles = [
        Line2D([0], [0], color='dimgray', linewidth=4, alpha=0.35,
               label='Cariaco bin mean (monthly)'),
        Line2D([0], [0], color='dimgray', linestyle='dashed', linewidth=2,
               label='Model bin aggregate'),
        Line2D([0], [0], color='dimgray', linewidth=1.2,
               label='Cariaco monthly min–max'),
    ]
    auto_h, auto_l = ax.get_legend_handles_labels()
    ax.legend(auto_h + extra_handles,
              auto_l + [h.get_label() for h in extra_handles],
              loc='best', fontsize=9)

    # Axes + title
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('ESD (µm)')
    ax.set_ylabel('Biomass density (mmol N m$^{-3}$ per log$_{10}$ µm)')

    if title is None:
        title = (f'Normalised biomass spectrum — model vs Cariaco '
                 f'({regime}-regime)')
        if show_zoo:
            title += '\nObs zoo bins via net-tow subtraction (>200 − >500)'
            if show_octave:
                title += '; model community on octave-in-ESD bins'
        else:
            title += '\nphyto-only'
            if show_octave:
                title += '; model community on octave-in-ESD bins'
    ax.set_title(title)
    plt.tight_layout()
    return ax