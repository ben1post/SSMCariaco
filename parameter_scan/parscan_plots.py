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
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, NullFormatter


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


# =============================================================================
# REGIME DIAGNOSTIC — shared conventions
# =============================================================================
# Single regime colour scheme used across every regime figure (forcing,
# diagnostic). Upwelling = blue (cool, nutrient-rich), relaxed = red (warm).
# NOTE: the two prototype cells disagreed on this (the diagnostic used
# upwelling=blue, the forcing figure used upwelling=red); standardised here on
# the diagnostic's scheme. Flip the two hex values to invert globally.
REGIME_COLORS = {'upwelling': '#2b6cb0', 'relaxed': '#c0392b'}
DEAD_MARKERS  = {'upwelling': 'x',       'relaxed': '+'}

# Sieburth phyto bands and coarse zoo bands for the spectrum panels.
PHYTO_BANDS = [(0.2, 2, 'Pico', '#dbe9f6'),
               (2, 20, 'Nano', '#bcd6ef'),
               (20, 200, 'Micro', '#9cc3e8')]
ZOO_BANDS   = [(4, 200, '<200', '#fde2cc'),
               (200, 500, '200–500', '#fbc49b'),
               (500, 2500, '>500', '#f8a76b')]

# Per-variant colours for the supply-sweep / stability figure.
VARIANT_COLORS = {'Type II': '#c44e2c', 'Type III': '#3a6ea5',
                  'baseline': '#c44e2c', 'baseline+fish': '#8b5a00'}


def _bar_dist(ax, x, values, color, hatch=None, width=0.4):
    """Bar at the median + whisker spanning the 10th-90th percentile.

    One visual grammar for both a model *orbit* distribution (the limit-cycle
    tail) and an obs *monthly* distribution: solid bar = median, dots+line =
    10-90% spread. Empty input prints a small 'no data' note instead.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        ax.text(x, 0.02, 'no\ndata', transform=ax.get_xaxis_transform(),
                ha='center', va='bottom', fontsize=6, color='grey')
        return
    lo, hi = np.percentile(v, [10, 90])
    ax.bar(x, np.median(v), width, color=color, alpha=0.9, hatch=hatch,
           edgecolor='k', linewidth=0.4, zorder=2)
    ax.plot([x, x], [lo, hi], color='k', lw=0.5, zorder=4)
    ax.scatter([x, x], [lo, hi], color='k', s=5, zorder=5)


def _spectrum_axis_limits(alive, dead_present, dead_drop=0.5, pad=0.2):
    """Tight log y-limits for a spectrum panel + a y-position for dead-class
    markers placed just below the live data.

    Sizing the axis to the live data range (not to decade floor/ceil) keeps the
    live lines filling the panel; dead markers go `dead_drop` decades below the
    live minimum so they read clearly without dragging the axis down to a
    far-away decade floor (which previously squashed the lines together).
    Returns (ylo, yhi, y_dead); y_dead is None when nothing is dead.
    """
    alive = np.asarray(alive, dtype=float)
    if alive.size == 0:
        return 1e-6, 1.0, 1e-6
    lo, hi = float(np.log10(alive.min())), float(np.log10(alive.max()))
    if dead_present:
        y_dead = 10 ** (lo - dead_drop)
        ylo = 10 ** (lo - dead_drop - pad)
    else:
        y_dead = None
        ylo = 10 ** (lo - pad)
    return ylo, 10 ** (hi + pad), y_dead


def _draw_spectrum(ax, esd, spec_by_regime, bands, xlim, title,
                   regimes=('upwelling', 'relaxed'), dead=1e-10):
    """Tail-mean size spectrum per regime, log-log, with size bands and
    dead-class markers placed dynamically just below the live data."""
    esd = np.asarray(esd)
    for lo, hi, lab, c in bands:
        ax.axvspan(lo, hi, color=c, alpha=0.45, zorder=0)
        ax.text(np.sqrt(lo * hi), 0.97, lab, transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=7, color='0.35')
    alive, any_dead = [], False
    for r in regimes:
        y = np.asarray(spec_by_regime[r], dtype=float)
        a = y >= dead
        any_dead = any_dead or bool((~a).any())
        ax.plot(esd[a], y[a], 'o-', color=REGIME_COLORS[r], ms=4, lw=1,
                label=r, zorder=3)
        alive.append(y[a])
    ylo, yhi, y_dead = _spectrum_axis_limits(np.concatenate(alive), any_dead)
    ax.set(xscale='log', yscale='log', xlim=xlim, ylim=(ylo, yhi),
           title=title, xlabel='ESD (µm)', ylabel='biomass (mmol N m⁻³)')
    for r in regimes:
        d = np.asarray(spec_by_regime[r], dtype=float) < dead
        if d.any():
            ax.plot(esd[d], np.full(d.sum(), y_dead), DEAD_MARKERS[r],
                    color=REGIME_COLORS[r], ms=8, mew=1.8, alpha=0.8,
                    zorder=6, clip_on=False)
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.legend(fontsize=8, loc='lower left')


# =============================================================================
# FIG 0 — model equations sheet (generic mathtext renderer)
# =============================================================================
def plot_equation_sheet(blocks, title=None, figsize=(12, 7.5),
                        x=0.05, y_top=0.93, line_gap=0.052):
    """Render a sheet of (mathtext) lines — the per-construct equation figure.

    Content-driven so each notebook supplies its own lines: spell the baseline
    out fully, then for derived constructs change only the `blocks` list.

    Parameters
    ----------
    blocks : list of (text, opts) tuples
        `text` is a (mathtext) string; `opts` a dict with any of:
          'size'   (float, default 14), 'weight' ('normal'|'bold', default
          'normal'), 'color' (default 'k'), 'gap' (extra blank-line spacing
          before this block, default 0 — use it to set off section headers).
    title : str or None — bold title at the top.
    """
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor('white')
    # Full-figure invisible axes: the inline backend skips rendering a figure
    # with zero axes (it shows the '<Figure ... with 0 Axes>' repr instead), so
    # add one and lay the text out in figure coordinates on top of it.
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    y = y_top
    if title is not None:
        fig.text(x, y, title, fontsize=13, weight='bold', color='#1a2733',
                 va='top')
        y -= line_gap * 1.7
    for text, opts in blocks:
        y -= line_gap * opts.get('gap', 0)
        fig.text(x, y, text, fontsize=opts.get('size', 14),
                 weight=opts.get('weight', 'normal'),
                 color=opts.get('color', 'k'), va='top')
        y -= line_gap * max(opts.get('size', 14) / 14.0, 0.9)
    return fig


# =============================================================================
# FIG 1 — parameter table (generic styled-table renderer)
# =============================================================================
def plot_parameter_table(rows,
                         col_labels=('Symbol / component', 'Form / value', 'Source'),
                         title=None, figsize=(11.5, 5),
                         col_widths=(0.22, 0.54, 0.24),
                         header_color='#33485e', alt_color='#eef2f6'):
    """Render a styled parameter table. Content-driven: pass `rows` (a list of
    same-length tuples). Header bar + alternating row shading applied for
    legibility."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    tbl = ax.table(cellText=rows, colLabels=col_labels, cellLoc='left',
                   colLoc='left', loc='center', colWidths=list(col_widths))
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10.5)
    tbl.scale(1, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color='w', weight='bold')
        elif r % 2:
            cell.set_facecolor(alt_color)
    if title:
        ax.set_title(title, fontsize=12.5, weight='bold', pad=14)
    plt.tight_layout()
    return fig


# =============================================================================
# FIG 2 — regime forcing contrast (upwelling vs relaxed)
# =============================================================================
def plot_regime_forcing(monthly_by_regime, regimes=('upwelling', 'relaxed'),
                        fn_col='FN_mmolN_m2_d', de_col='depth_cutoff',
                        figsize=(13, 4.2),
                        title='Regime forcing contrast — HPLC-resolved months '
                              '(medians = model forcing)'):
    """Boxplot F_N, d_e, and volumetric supply F_N/d_e per regime.

    monthly_by_regime : dict {regime: monthly_df from load_cariaco_targets}.
    The per-month spread is shown; the medians are the values the model is
    forced with.
    """
    fig, ax = plt.subplots(1, 3, figsize=figsize)

    def _box(a, series_by_regime, ttl, ylab, invert=False):
        vals = [np.asarray(series_by_regime[r], float) for r in regimes]
        vals = [v[np.isfinite(v)] for v in vals]
        bp = a.boxplot(vals, patch_artist=True, widths=0.5, showfliers=False)
        for patch, r in zip(bp['boxes'], regimes):
            patch.set_facecolor(REGIME_COLORS[r])
            patch.set_alpha(0.55)
        for med in bp['medians']:
            med.set_color('k')
            med.set_linewidth(2)
        a.set(title=ttl, ylabel=ylab)
        a.set_xticks(range(1, len(regimes) + 1))
        a.set_xticklabels(regimes)
        if invert:
            a.set_ylim(bottom=0)
            a.invert_yaxis()       # surface (0) at top, depth downward

    fn = {r: monthly_by_regime[r][fn_col].dropna().values for r in regimes}
    de = {r: monthly_by_regime[r][de_col].dropna().values for r in regimes}
    vsup = {r: (monthly_by_regime[r][fn_col] / monthly_by_regime[r][de_col])
                .dropna().values for r in regimes}

    _box(ax[0], fn,   'Nutrient supply F_N',        'F_N  (mmol N m⁻² d⁻¹)')
    _box(ax[1], de,   'Box / euphotic depth d_e',   'd_e  (m)', invert=True)
    _box(ax[2], vsup, 'Volumetric supply F_N / d_e', 'mmol N m⁻³ d⁻¹')

    fig.suptitle(title, fontsize=12.5, weight='bold')
    plt.tight_layout()
    return fig


# =============================================================================
# FIG 4 — regime diagnostic (model orbit vs obs, upwelling vs relaxed)
# =============================================================================
def plot_regime_diagnostic(model_by_regime, obs_by_regime, phyto_esd, zoo_esd,
                           regimes=('upwelling', 'relaxed'),
                           title='R0 baseline diagnostic', figsize=(17, 9.3)):
    """Model-vs-obs regime diagnostic, consuming process_regime_run (model) and
    process_regime_obs (obs) outputs.

    Panels — top: ΣP/ΣZ stability traces (+CV), tail-mean phyto spectrum,
    tail-mean zoo spectrum. Bottom: mean cell size, ΣP biomass, phyto bins,
    small zoo (model only), large zoo (vs obs). Model orbit distributions and
    obs monthly distributions are both shown as median + 10-90% (solid = model,
    hatched = obs). Designed for the two-regime upwelling/relaxed contrast.
    """
    ramp3 = {'upwelling': plt.cm.Blues([.45, .65, .88]),
             'relaxed':   plt.cm.Reds([.45, .65, .88])}
    ramp4 = {'upwelling': plt.cm.Blues([.35, .55, .75, .95]),
             'relaxed':   plt.cm.Reds([.35, .55, .75, .95])}

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 6, hspace=0.6, wspace=0.55)
    ax_stab = fig.add_subplot(gs[0, 0:2])
    ax_ps   = fig.add_subplot(gs[0, 2:4])
    ax_zs   = fig.add_subplot(gs[0, 4:6])
    ax_mc   = fig.add_subplot(gs[1, 0])
    ax_bio  = fig.add_subplot(gs[1, 1])
    ax_pb   = fig.add_subplot(gs[1, 2:4])
    ax_zsm  = fig.add_subplot(gs[1, 4])
    ax_zlg  = fig.add_subplot(gs[1, 5])

    # --- stability traces: ΣP solid, ΣZ dashed ---
    for r in regimes:
        m = model_by_regime[r]
        ax_stab.plot(m['t'], m['sumP'], color=REGIME_COLORS[r], lw=0.8,
                     label=f"{r} ΣP (CV={m['cv']:.2f})")
        ax_stab.plot(m['t'], m['sumZ'], color=REGIME_COLORS[r], lw=0.8, ls='--')
    ax_stab.set(title='Stability — ΣP solid, ΣZ dashed',
                xlabel='day', ylabel='biomass (mmol N m⁻³)')
    ax_stab.legend(fontsize=7)

    # --- tail-mean spectra ---
    _draw_spectrum(ax_ps, phyto_esd, {r: model_by_regime[r]['Pspec'] for r in regimes},
                   PHYTO_BANDS, (0.4, 250), 'Tail-mean phyto spectrum', regimes)
    _draw_spectrum(ax_zs, zoo_esd, {r: model_by_regime[r]['Zspec'] for r in regimes},
                   ZOO_BANDS, (4, 2500), 'Tail-mean zoo spectrum', regimes)

    # --- mean cell size + ΣP biomass: model (orbit) vs obs (monthly) ---
    mo_x = {'upwelling': (0, 0.55), 'relaxed': (1.5, 2.05)}   # (model_x, obs_x)
    panels = [(ax_mc, 'mcs_tail', 'mcs', 'mean cell size', 'µm'),
              (ax_bio, 'sumP_tail', 'sumP', 'ΣP biomass', 'mmol N m⁻³')]
    for a, mkey, okey, ttl, yl in panels:
        for r in regimes:
            _bar_dist(a, mo_x[r][0], model_by_regime[r][mkey], REGIME_COLORS[r])
            _bar_dist(a, mo_x[r][1], obs_by_regime[r][okey], REGIME_COLORS[r],
                      hatch='////')
        a.set_xticks([0, 0.55, 1.5, 2.05])
        a.set_xticklabels(['m', 'o', 'm', 'o'], fontsize=7)
        a.set_title(ttl, fontsize=9)
        a.set_ylabel(yl, fontsize=8)
        a.grid(axis='y', ls='--', alpha=0.4)

    # --- phyto bins (Pico/Nano/Micro), model vs obs, per regime ---
    gc = {'upwelling': 0, 'relaxed': 3.6}
    bdx = [-0.9, 0, 0.9]
    for r in regimes:
        for k in range(3):
            xb = gc[r] + bdx[k]
            _bar_dist(ax_pb, xb - 0.18, model_by_regime[r]['phyto_bins_tail'][k],
                      ramp3[r][k], width=0.3)
            _bar_dist(ax_pb, xb + 0.18, obs_by_regime[r]['phyto_bins'][k],
                      ramp3[r][k], hatch='////', width=0.3)
    ax_pb.set_xticks([gc[r] + d for r in regimes for d in bdx])
    ax_pb.set_xticklabels(['Pi', 'Na', 'Mi'] * 2, fontsize=7)
    ax_pb.set_title('phyto bins', fontsize=9)
    ax_pb.set_ylabel('mmol N m⁻³', fontsize=8)
    ax_pb.grid(axis='y', ls='--', alpha=0.4)

    # --- small zoo bands (model only) ---
    zsp = {'upwelling': (0, 0.6), 'relaxed': (1.6, 2.2)}
    band_keys = list(model_by_regime[regimes[0]]['zoo_band_tail'].keys())
    for r in regimes:
        for j, bk in enumerate(band_keys):
            _bar_dist(ax_zsm, zsp[r][j], model_by_regime[r]['zoo_band_tail'][bk],
                      ramp4[r][j], width=0.45)
    ax_zsm.set_xticks([zsp[r][j] for r in regimes for j in range(len(band_keys))])
    ax_zsm.set_xticklabels(band_keys * 2, fontsize=6, rotation=30, ha='right')
    ax_zsm.set_title('small zoo — model only', fontsize=8.5)
    ax_zsm.set_ylabel('mmol N m⁻³', fontsize=8)
    ax_zsm.grid(axis='y', ls='--', alpha=0.4)

    # --- large zoo (cumulative >thr), model vs obs ---
    zlp = {'upwelling': 0, 'relaxed': 2.4}
    ldx = [-0.5, 0.5]
    for r in regimes:
        cum_keys = list(model_by_regime[r]['zoo_cum_tail'].keys())   # gt200, gt500
        obs_vals = list(obs_by_regime[r]['zoo'].values())            # >200, >500
        for k, ck in enumerate(cum_keys):
            xb = zlp[r] + ldx[k]
            _bar_dist(ax_zlg, xb - 0.16, model_by_regime[r]['zoo_cum_tail'][ck],
                      ramp4[r][k + 2], width=0.28)
            ov = obs_vals[k] if k < len(obs_vals) else []
            _bar_dist(ax_zlg, xb + 0.16, ov, ramp4[r][k + 2],
                      hatch='////', width=0.28)
    ax_zlg.set_xticks([zlp[r] + d for r in regimes for d in ldx])
    ax_zlg.set_xticklabels(['>200', '>500', '>200', '>500'], fontsize=6.5)
    ax_zlg.set_title('large zoo — vs obs', fontsize=8.5)
    ax_zlg.set_ylabel('mmol N m⁻³', fontsize=8)
    ax_zlg.grid(axis='y', ls='--', alpha=0.4)

    # --- central regime colour key + figure legend ---
    fig.text(0.495, 0.485, regimes[0].upper(), color=REGIME_COLORS[regimes[0]],
             fontsize=16, weight='bold', ha='right', va='center')
    fig.text(0.505, 0.485, regimes[1].upper(), color=REGIME_COLORS[regimes[1]],
             fontsize=16, weight='bold', ha='left', va='center')
    fig.text(0.5, 0.45, 'colour = regime   ·   solid = model, hatched = obs   ·   '
             'bar = median, dots = 10–90%   ·   × / + = dead phyto class',
             fontsize=8.5, color='0.4', ha='center', va='center')
    fig.legend(handles=[Patch(facecolor='0.6', label='model (median · 10–90%)'),
                        Patch(facecolor='0.6', hatch='////',
                              label='obs (median · 10–90%)')],
               loc='upper center', ncol=2, fontsize=8.5,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(title, fontsize=14, weight='bold', y=1.05)
    return fig


# =============================================================================
# FIG 5 — supply-sweep stability comparison (Type II vs Type III)
# =============================================================================
def plot_stability_scan(variants, fn_values, regime_fn=None,
                        metric_key='cv', metric_label='CV(ΣP) over tail',
                        metric_threshold=0.1, cell_key='mean_cell',
                        figsize=(13, 4.6),
                        title='Supply sweep — stability and mean cell size'):
    """Two-panel F_N sweep: (left) a stability metric, (right) mean cell size,
    one line per model variant.

    Parameters
    ----------
    variants : dict {variant_label: {metric_key: 1D array, cell_key: 1D array}}
        e.g. {'Type II': {'cv': [...], 'mean_cell': [...]},
              'Type III': {'cv': [...], 'mean_cell': [...]}}.
        `metric_key` chooses the left-panel quantity — 'cv' for the IVP orbit
        amplitude, or e.g. 'max_eig' for the stability-solver dominant
        eigenvalue real part (set metric_label/threshold to match).
    fn_values : 1D array of F_N values (shared x-axis).
    regime_fn : dict {regime: F_N} or None — vertical reference lines.
    """
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    markers = ['o-', 's-', '^-', 'd-']
    for i, (label, data) in enumerate(variants.items()):
        c = VARIANT_COLORS.get(label, None)
        mk = markers[i % len(markers)]
        ax[0].plot(fn_values, data[metric_key], mk, color=c, label=label)
        ax[1].plot(fn_values, data[cell_key], mk, color=c, label=label)

    if metric_threshold is not None:
        ax[0].axhline(metric_threshold, ls=':', color='grey')
    ax[0].set(title=f'Stability: {metric_label}',
              xlabel='F_N (mmol N m⁻² d⁻¹)', ylabel=metric_label)
    ax[1].set(title='Mean cell size vs supply',
              xlabel='F_N (mmol N m⁻² d⁻¹)', ylabel='mean cell (µm)')

    if regime_fn:
        for r, fn in regime_fn.items():
            for a in ax:
                a.axvline(fn, ls='--', color=REGIME_COLORS.get(r, 'grey'),
                          alpha=0.6)
            ax[1].annotate(r, (fn, ax[1].get_ylim()[1] * 0.95), fontsize=8,
                           rotation=90, va='top',
                           color=REGIME_COLORS.get(r, 'grey'))
    for a in ax:
        a.legend(fontsize=9)
        a.grid(alpha=0.3)
    fig.suptitle(title, fontsize=12.5, weight='bold')
    plt.tight_layout()
    return fig


# =============================================================================
# FIG 6 — stability diagnostic (fsolve focus vs IVP orbit + eigenvalue spectrum)
# =============================================================================
def plot_stability_diagnostic(stab_by_regime, phyto_esd,
                              regimes=('upwelling', 'relaxed'),
                              figsize=(12, 8), dead=1e-10):
    """Per regime, two stacked panels:
      (top)  fsolve steady-state phyto spectrum (solid) vs IVP tail-mean
             (dashed) — the unstable focus vs the orbit average it sits in.
      (bottom) Jacobian eigenvalue Re/Im scatter, split at Re=0, annotated
             with the stability label, max Re(λ), #positive, #complex pairs.

    `stab_by_regime[r]` must carry: 'stab' (the hook.get_results() dict, with
    'eigenvalues_real'/'eigenvalues_imag'/'stability'/'max_eigenvalue_real'/
    'n_positive_eigenvalues'/'n_complex_pairs'), 'P_ss', and 'P_tail'.
    """
    esd = np.asarray(phyto_esd)
    n = len(regimes)
    fig, ax = plt.subplots(2, n, figsize=figsize, squeeze=False)

    for j, r in enumerate(regimes):
        d = stab_by_regime[r]
        s = d['stab']
        col = REGIME_COLORS[r]
        mk = DEAD_MARKERS[r]

        # ---- top: steady-state vs orbit-average phyto spectrum ----
        a = ax[0, j]
        for lo, hi, lab, c in PHYTO_BANDS:
            a.axvspan(lo, hi, color=c, alpha=0.45, zorder=0)
            a.text(np.sqrt(lo * hi), 0.97, lab, transform=a.get_xaxis_transform(),
                   ha='center', va='top', fontsize=7, color='0.35')
        ss = np.asarray(d['P_ss'], float)
        tl = np.asarray(d['P_tail'], float)
        alive = np.concatenate([ss[ss >= dead], tl[tl >= dead]])
        any_dead = bool((ss < dead).any() or (tl < dead).any())
        ylo, yhi, y_dead = _spectrum_axis_limits(alive, any_dead)
        a.plot(esd[ss >= dead], ss[ss >= dead], 'o-', color=col, ms=4, lw=1.2,
               label='fsolve steady state', zorder=3)
        a.plot(esd[tl >= dead], tl[tl >= dead], 's--', color=col, ms=4, lw=1,
               alpha=0.55, label='IVP tail-mean (orbit)', zorder=3)
        # dead-class markers just below the live data (bold = ss, faint = orbit)
        if (ss < dead).any():
            a.plot(esd[ss < dead], np.full((ss < dead).sum(), y_dead), mk, color=col,
                   ms=8, mew=1.8, zorder=6, clip_on=False)
        if (tl < dead).any():
            a.plot(esd[tl < dead], np.full((tl < dead).sum(), y_dead), mk, color=col,
                   ms=8, mew=1.2, alpha=0.45, zorder=6, clip_on=False)
        a.set(xscale='log', yscale='log', xlim=(0.4, 250), ylim=(ylo, yhi),
              xlabel='ESD (µm)', ylabel='biomass (mmol N m⁻³)')
        a.set_title(f'{r} — phyto spectrum', fontsize=10)
        a.legend(fontsize=7, loc='lower left')

        # ---- bottom: eigenvalue spectrum, zoomed to the imaginary axis ----
        # Fast-decaying modes (deep-negative Re) carry no stability information
        # and would squash the relevant modes against Re=0, so we focus a window
        # around zero and report the off-scale count.
        b = ax[1, j]
        re = np.asarray(s['eigenvalues_real'])
        im = np.asarray(s['eigenvalues_imag'])
        max_re = float(s['max_eigenvalue_real'])
        half = max(abs(max_re), 0.02) * 3.0
        in_view = re >= -half
        n_off = int((~in_view).sum())

        b.axvspan(0, half, color='#c0392b', alpha=0.06, zorder=0)   # unstable side
        b.axvline(0, color='k', lw=0.9)
        st = re < 0
        b.scatter(re[st & in_view], im[st & in_view], s=18, color='#3a6ea5',
                  edgecolor='k', lw=0.3, label='stable  (Re<0)', zorder=3)
        b.scatter(re[~st], im[~st], s=26, color='#c0392b', edgecolor='k',
                  lw=0.3, label='unstable  (Re≥0)', zorder=3)
        # star the dominant eigenvalue + its oscillation period
        kmax = int(np.argmax(re))
        b.scatter([re[kmax]], [im[kmax]], s=150, marker='*', color='gold',
                  edgecolor='k', lw=0.6, zorder=5)
        period = 2 * np.pi / abs(im[kmax]) if abs(im[kmax]) > 1e-9 else np.inf
        per_txt = f", period ≈ {period:.0f} d" if np.isfinite(period) else ""
        b.annotate(f'dominant: Re={max_re:+.4f}{per_txt}', (re[kmax], im[kmax]),
                   textcoords='offset points', xytext=(7, 6), fontsize=7.5)

        b.set_xlim(-half, half)
        b.set(xlabel='Re(λ): growth rate [d⁻¹]   (>0 → unstable)',
              ylabel='Im(λ): oscillation [rad d⁻¹]')
        ttl = (f"{r} — {s['stability']}  "
               f"({s['n_positive_eigenvalues']} pos, {s['n_complex_pairs']} complex pairs)")
        if n_off:
            ttl += f"  · {n_off} fast modes off-scale"
        b.set_title(ttl, fontsize=9)
        b.legend(fontsize=7, loc='upper left')
        b.grid(alpha=0.3)

    fig.suptitle('Stability — fsolve fixed point (seeded from IVP tail-mean) '
                 'vs the orbit it sits in', fontsize=13, weight='bold')
    plt.tight_layout()
    return fig