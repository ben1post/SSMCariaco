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
                      best, default=None, figsize=(9, 7)):
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
    """
    fig, ax = plt.subplots(figsize=figsize)

    vmax = np.nanpercentile(cost_grid, 90)
    im = ax.pcolormesh(vals2, vals1, cost_grid,
                       cmap='viridis_r', vmin=0, vmax=vmax, shading='auto')

    # Contour lines for good-fit regions
    X, Y = np.meshgrid(vals2, vals1)
    levels = [0.3, 0.5, 1.0, 2.0]
    valid_levels = [l for l in levels if l <= np.nanmax(cost_grid)]
    if valid_levels:
        cs = ax.contour(X, Y, cost_grid, levels=valid_levels,
                        colors='white', linewidths=1.0, alpha=0.7)
        ax.clabel(cs, fmt='%.1f', fontsize=8)

    # Best fit
    ax.plot(best['val2'], best['val1'], '*', color='red', markersize=18,
            markeredgecolor='white', markeredgewidth=1.2,
            label=f"Best fit (cost={best['cost']:.3f})")

    # Default
    if default is not None:
        ax.plot(default['val2'], default['val1'], 'D', color='orange',
                markersize=10, markeredgecolor='white', markeredgewidth=1.0,
                label='Default params')

    fig.colorbar(im, ax=ax, label='Cost (NRMSRE)', shrink=0.85)
    ax.set_xlabel(p2_label, fontsize=12)
    ax.set_ylabel(p1_label, fontsize=12)
    ax.set_title('2D Parameter Scan: Fit to CARIACO Observations\n'
                 'Cost = Normalized RMSRE across targets', fontsize=13)
    ax.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    # fig.savefig('cost_heatmap.pdf', bbox_inches='tight')
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