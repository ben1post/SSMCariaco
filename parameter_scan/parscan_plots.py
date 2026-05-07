"""
Parameter Scan Plots
====================
Plotting utilities for CARIACO 2D parameter scans:
  - Cost heatmap with best-fit and default markers
  - Model vs. observation bar comparison
  - Model vs. observation boxplots (showing monthly variance)
  - Best-fit numerical summary table

All plot functions return the figure object so it can be saved by the caller.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# COLOR PALETTES BY TARGET TYPE
# =============================================================================
# Each type maps to a list of shades (light -> dark) indexed by the target's
# position within its type group. Extend this dict when adding new types.
TYPE_COLOR_PALETTES = {
    'phyto':    ['#95d5b2', '#52b788', '#2d6a4f'],   # greens: pico -> micro
    'zoo':      ['#f48c06', '#e85d04'],              # oranges: gt200 -> gt500
    'nutrient': ['#6a4c93'],                         # purple
    'detritus': ['#7b4b2a'],                         # brown
    'export':   ['#808080'],                         # grey
}

TYPE_UNITS = {
    'phyto':    'mmol N m⁻³',
    'zoo':      'mmol N m⁻³',
    'nutrient': 'mmol N m⁻³',
    'detritus': 'mmol N m⁻³',
    'export':   'mmol N m⁻² d⁻¹',
}

def _assign_colors(bin_definitions):
    """Return one color per bin definition, based on type + order within type."""
    colors = []
    type_counters = {}
    for b in bin_definitions:
        t = b['type']
        palette = TYPE_COLOR_PALETTES.get(t, ['#888888'])
        idx = type_counters.get(t, 0)
        colors.append(palette[idx % len(palette)])
        type_counters[t] = idx + 1
    return colors


def _group_by_type(bin_definitions):
    """Return list of (type_name, [indices_into_bin_definitions]) in order of first appearance."""
    groups = {}
    order = []
    for i, b in enumerate(bin_definitions):
        t = b['type']
        if t not in groups:
            groups[t] = []
            order.append(t)
        groups[t].append(i)
    return [(t, groups[t]) for t in order]


# =============================================================================
# 1. COST HEATMAP
# =============================================================================
def plot_cost_heatmap(cost_grid, vals1, vals2, p1_label, p2_label,
                      best, default=None, stable_mask=None, figsize=(9, 7)):
    """
    2D cost heatmap with contours, best-fit marker, and optional default marker.

    Parameters
    ----------
    cost_grid : np.ndarray, shape (n1, n2)
        Cost values; NaN entries are masked.
    vals1, vals2 : arrays
        Parameter values for dim1 (y-axis) and dim2 (x-axis).
    p1_label, p2_label : str
        Axis labels.
    best : dict
        Output of find_best_fit().
    default : dict or None
        Optional: {'val1': float, 'val2': float} to mark default parameters.
    stable_mask : np.ndarray of bool or None, optional
        Same shape as `cost_grid`; True where the cell is a stable steady
        state. If provided, a black contour outlining the stable region is
        overlaid on the heatmap.
    """
    fig, ax = plt.subplots(figsize=figsize)

    vmax = np.nanpercentile(cost_grid, 90)
    im = ax.pcolormesh(vals2, vals1, cost_grid,
                       cmap='viridis_r', vmin=0, vmax=vmax, shading='auto')

    X, Y = np.meshgrid(vals2, vals1)
    levels = [0.3, 0.5, 1.0, 2.0]
    valid_levels = [l for l in levels if l <= np.nanmax(cost_grid)]
    if valid_levels:
        cs = ax.contour(X, Y, cost_grid, levels=valid_levels,
                        colors='white', linewidths=1.0, alpha=0.7)
        ax.clabel(cs, fmt='%.1f', fontsize=8)

    if stable_mask is not None:
        unstable_mask = ~stable_mask
        unstable_overlay = np.where(unstable_mask, 1.0, np.nan)
        ax.pcolor(vals2, vals1, unstable_overlay,
                  hatch='////', alpha=0.0, shading='auto')

    ax.plot(best['val2'], best['val1'], '*', color='red', markersize=18,
            markeredgecolor='white', markeredgewidth=1.2,
            label=f"Best fit (cost={best['cost']:.3f})")

    if default is not None:
        ax.plot(default['val2'], default['val1'], 'D', color='orange',
                markersize=10, markeredgecolor='white', markeredgewidth=1.0,
                label='Default params')

    fig.colorbar(im, ax=ax, label='Cost (NRMSRE)', shrink=0.85)
    ax.set_xlabel(p2_label, fontsize=12)
    ax.set_ylabel(p1_label, fontsize=12)
    ax.set_title('2D Parameter Scan: Fit to CARIACO Observations\n'
                 'Cost = Normalized RMSRE across targets', fontsize=13)
    handles, labels = ax.get_legend_handles_labels()
    if stable_mask is not None:
        from matplotlib.patches import Patch
        handles.append(Patch(facecolor='white', edgecolor='black',
                             hatch='////', label='Unstable'))
        labels.append('Unstable')
    ax.legend(handles, labels, loc='upper right', fontsize=10)
    plt.tight_layout()
    return fig


# =============================================================================
# 1B. SPECTRUM COMPOSITION MAP (RGB)
# =============================================================================
def plot_spectrum_composition_map(
    model_grid, obs_vec, bin_definitions,
    vals1, vals2, p1_label, p2_label,
    *,
    type_filter='phyto',
    best=None, spectrum_best=None, default=None,
    contour_levels=(0.05, 0.10, 0.20, 0.40),
    gamma=0.5, figsize=(10, 7),
):
    """
    RGB composition map of one target type across a 2D parameter scan.

    Each pixel of the (n1, n2) parameter plane is coloured by the model's
    *relative* composition across the three targets of ``type_filter``.
    For ``type_filter='phyto'`` the channels are Pico (R), Nano (G),
    Micro (B). Pixel brightness is gamma-corrected closeness of the
    modelled composition to the observed composition (bright = match).

    Overlays:
      - White contours of the relative-composition distance
        (the value computed by ``compute_cost_relative_spectrum``).
      - Optional NRMSRE best-fit marker (yellow star).
      - Optional spectrum-only best-fit marker (magenta plus).
      - Optional default-parameter marker (cyan diamond).

    Parameters
    ----------
    model_grid : np.ndarray, shape (n1, n2, n_targets)
        Per-cell aggregated target vectors from ``compute_cost_grid``.
    obs_vec : np.ndarray, shape (n_targets,)
    bin_definitions : list of dict
    vals1, vals2 : arrays
        Parameter values along dim1 (y-axis) and dim2 (x-axis), same
        convention as ``plot_cost_heatmap``.
    p1_label, p2_label : str
    type_filter : str, optional
        Target type whose composition drives the RGB map. Must have
        exactly 3 targets in ``bin_definitions``. Default ``'phyto'``.
    best : dict or None
        Output of ``find_best_fit`` on the NRMSRE cost grid; plotted
        as a yellow star. Optional.
    spectrum_best : dict or None
        Output of ``find_best_fit`` on the spectrum-only cost grid;
        plotted as a magenta plus. Optional.
    default : dict or None
        ``{'val1': float, 'val2': float}`` for default-parameter marker.
        Optional.
    contour_levels : tuple of float, optional
        Levels for the distance contour overlay. Range 0..sqrt(2).
    gamma : float, optional
        Gamma exponent for the brightness encoding. Lower flattens
        contrast; default 0.5 follows the prototype.
    figsize : tuple, optional

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    from matplotlib.patches import Patch

    idx = [i for i, b in enumerate(bin_definitions)
           if b['type'] == type_filter]
    if len(idx) != 3:
        raise ValueError(
            f"plot_spectrum_composition_map needs exactly 3 targets of "
            f"type '{type_filter}' in bin_definitions, found {len(idx)}."
        )
    labels = [bin_definitions[i]['label'] for i in idx]

    # Relative composition per cell over the filtered targets
    sub = np.asarray(model_grid)[:, :, idx]
    totals = sub.sum(axis=-1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        rel = np.where(totals > 0, sub / totals, np.nan)

    # Observed composition (constant across the plane)
    obs_sub = np.asarray(obs_vec, dtype=float)[idx]
    obs_rel = obs_sub / obs_sub.sum()

    # Distance to obs composition per cell
    dist = np.linalg.norm(rel - obs_rel, axis=-1)

    # Brightness: bright where dist is small. Gamma-corrected, normalised
    # to the 95th percentile to keep the dynamic range bounded.
    dist_max = np.nanpercentile(dist, 95)
    if not np.isfinite(dist_max) or dist_max <= 0:
        brightness = np.zeros_like(dist)
    else:
        brightness = (1.0 - np.clip(dist, 0, dist_max) / dist_max) ** gamma

    rgb = np.nan_to_num(rel * brightness[..., None], nan=0.0)
    rgb = np.clip(rgb, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(
        rgb, origin='lower', aspect='auto',
        extent=[vals2.min(), vals2.max(), vals1.min(), vals1.max()],
        zorder=1,
    )

    X, Y = np.meshgrid(vals2, vals1)
    dist_max_overall = np.nanmax(dist) if np.any(np.isfinite(dist)) else 0.0
    valid_levels = [l for l in contour_levels if l <= dist_max_overall]
    if valid_levels:
        cs = ax.contour(
            X, Y, dist, levels=valid_levels,
            colors='white', linewidths=1.2, alpha=0.85, zorder=3,
        )
        ax.clabel(cs, fmt='%.2f', fontsize=8)

    legend_handles = [
        Patch(facecolor='red',   label=f'{labels[0]} dominates'),
        Patch(facecolor='green', label=f'{labels[1]} dominates'),
        Patch(facecolor='blue',  label=f'{labels[2]} dominates'),
        Patch(facecolor='white', edgecolor='grey',
              label='Contour: rel-comp distance'),
    ]

    if best is not None:
        ax.plot(
            best['val2'], best['val1'], '*',
            color='yellow', markersize=18, markeredgecolor='black',
            markeredgewidth=1.0, zorder=5,
        )
        legend_handles.append(plt.Line2D(
            [], [], marker='*', linestyle='', color='yellow',
            markeredgecolor='black', markersize=14,
            label=f"NRMSRE best (cost={best['cost']:.3f})",
        ))
    if spectrum_best is not None:
        ax.plot(
            spectrum_best['val2'], spectrum_best['val1'], 'P',
            color='magenta', markersize=14, markeredgecolor='black',
            markeredgewidth=1.0, zorder=5,
        )
        legend_handles.append(plt.Line2D(
            [], [], marker='P', linestyle='', color='magenta',
            markeredgecolor='black', markersize=12,
            label=f"Spectrum best (dist={spectrum_best['cost']:.3f})",
        ))
    if default is not None:
        ax.plot(
            default['val2'], default['val1'], 'D',
            color='cyan', markersize=10, markeredgecolor='black',
            markeredgewidth=1.0, zorder=5,
        )
        legend_handles.append(plt.Line2D(
            [], [], marker='D', linestyle='', color='cyan',
            markeredgecolor='black', markersize=10,
            label='Default params',
        ))

    ax.set_xlabel(p2_label, fontsize=12)
    ax.set_ylabel(p1_label, fontsize=12)
    ax.set_title(
        f'{type_filter.capitalize()} composition map\n'
        f'R = {labels[0]}, G = {labels[1]}, B = {labels[2]}; '
        f'brightness = match to obs composition',
        fontsize=11,
    )
    ax.legend(handles=legend_handles, loc='lower right', fontsize=9)
    plt.tight_layout()
    return fig


# =============================================================================
# 2. MODEL vs OBS — BAR CHART
# =============================================================================
def plot_model_vs_obs_bars(model_vec, obs_vec, bin_definitions,
                           title_info='', figsize=(12, 5)):
    """
    Side-by-side bar comparison of observation means vs. model at best fit.

    Parameters
    ----------
    model_vec, obs_vec : array-like, shape (n_targets,)
    bin_definitions : list of dict
    title_info : str
        Extra string appended to the title (e.g. scan params + cost).
    """
    labels = [b['label'] for b in bin_definitions]
    colors = _assign_colors(bin_definitions)

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width/2, obs_vec, width, color=colors, alpha=0.5,
           edgecolor='black', linewidth=0.8, label='Obs (mean)')
    ax.bar(x + width/2, model_vec, width, color=colors,
           edgecolor='black', linewidth=0.8, label='Model (best fit)')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=15, ha='right')
    ax.set_ylabel('Biomass (mmol N m⁻³)', fontsize=11)
    title = 'Best-fit Model vs CARIACO Observations'
    if title_info:
        title += f'\n{title_info}'
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    # fig.savefig('model_vs_obs_bars.pdf', bbox_inches='tight')
    return fig


# =============================================================================
# 3. MODEL vs OBS — BOXPLOTS (monthly variance)
# =============================================================================
def plot_model_vs_obs_boxplots(model_vec, monthly_df, bin_definitions,
                               title_info='', figsize=None):
    """
    Boxplots of monthly observations with best-fit model values as stars.

    Panels are arranged by target type (phyto | zoo | nutrient | ...) in order
    of first appearance in bin_definitions. Panel widths scale with the number
    of targets in each group.

    Parameters
    ----------
    model_vec : array-like, shape (n_targets,)
    monthly_df : pd.DataFrame
        Monthly observations (from load_cariaco_targets), containing one column
        per target (named by bin_definitions[k]['column']).
    bin_definitions : list of dict
    title_info : str
        Extra string appended to the suptitle.
    """
    groups = _group_by_type(bin_definitions)
    colors = _assign_colors(bin_definitions)

    # Panel widths proportional to number of targets per group
    width_ratios = [len(idxs) for (_, idxs) in groups]
    if figsize is None:
        figsize = (3 + sum(width_ratios) * 1.5, 5)

    fig, axes = plt.subplots(1, len(groups), figsize=figsize,
                             gridspec_kw={'width_ratios': width_ratios})
    if len(groups) == 1:
        axes = [axes]

    star_kw = dict(marker='*', s=220, edgecolor='black',
                   linewidth=0.8, zorder=5)

    for ax, (type_name, idxs) in zip(axes, groups):
        cols = [bin_definitions[i]['column'] for i in idxs]
        labels = [bin_definitions[i]['label'] for i in idxs]
        group_colors = [colors[i] for i in idxs]
        group_model_vals = [model_vec[i] for i in idxs]

        # Boxplot of monthly values per target
        data = [monthly_df[c].dropna().values for c in cols]
        bp = ax.boxplot(data, positions=range(len(cols)), widths=0.5,
                        patch_artist=True, showmeans=True, showfliers=False,
                        meanprops=dict(marker='D', markerfacecolor='gray',
                                       markersize=5))
        for patch, color in zip(bp['boxes'], group_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.45)

        # Best-fit model values as stars
        ax.scatter(range(len(cols)), group_model_vals,
                   color=group_colors, **star_kw,
                   label='Model (best fit)' if ax is axes[0] else None)

        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(labels, fontsize=9, rotation=15, ha='right')
        ax.set_title(type_name.capitalize())
        ax.set_ylabel(TYPE_UNITS.get(type_name, ''), fontsize=9)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    axes[0].legend(fontsize=8)

    suptitle = 'CARIACO: Obs (monthly) vs Best-fit Model'
    if title_info:
        suptitle += f'  —  {title_info}'
    plt.suptitle(suptitle, fontsize=13, y=1.01)
    plt.tight_layout()
    # fig.savefig('model_vs_obs_boxplots.pdf', bbox_inches='tight')
    return fig


# =============================================================================
# 4. NUMERICAL SUMMARY
# =============================================================================
def summarize_best_fit(model_vec, obs_vec, labels, cost=None, verbose=True):
    """
    Build (and optionally print) a summary table of model vs. obs at best fit.

    Returns
    -------
    summary : pd.DataFrame
        Columns: Component, Obs_Mean, Model_BestFit, Ratio, Rel_Error_%
    """
    summary = pd.DataFrame({
        'Component':     labels,
        'Obs_Mean':      obs_vec,
        'Model_BestFit': model_vec,
        'Ratio':         model_vec / obs_vec,
        'Rel_Error_%':   100 * (model_vec - obs_vec) / obs_vec,
    })

    if verbose:
        print("\n" + "=" * 75)
        if cost is not None:
            print(f" BEST-FIT SUMMARY  |  Overall cost (NRMSRE): {cost:.4f}")
        else:
            print(" BEST-FIT SUMMARY")
        print("=" * 75)
        print(summary.to_string(index=False,
                                float_format=lambda x: f"{x:.5f}"))
        print("=" * 75)

    return summary