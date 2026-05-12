"""flux_plot.py — per-class phytoplankton flux decomposition for the Cariaco SSM.

Reads an XSO model-output Dataset (steady-state run, time index -1 by default)
and plots, per phyto size class i, all source/sink terms acting on P_i with
the steady-state biomass P_i* shown as the centre of each bar.

Two display modes (toggled via `mode=`):

  - 'biomass'    : log-y biomass axis [mmol N m-3]; bars centred at P_i*;
                   segment lengths are flux * dt (mass-equivalent contribution
                   over a notional unit time `dt`, default 1 day).
                   The size spectrum is directly visible as the centre line.

  - 'per_capita' : linear-y per-capita rate axis [d-1]; bars centred at 0;
                   segment lengths are per-capita source/sink rates.
                   Matches the analytical R*-style derivation framing
                   (Poulin & Franks 2010 Eq. 18-19; Taniguchi 2014 slope theorem).

Phyto grazing loss is decomposed into THREE predator groups per prey class:

  - smaller-than-optimal predators  (lighter shade)
  - the optimal predator             (mid orange — kernel argmax for that prey)
  - larger-than-optimal predators    (darker shade)

This preserves the directional information about which zoo class is killing
each P class without the visual clutter of 12 stacked sub-segments. The
classification follows the kernel itself (theta_opt, sigma_log) rather than
fixed ESD bins, so it remains meaningful across parameter sweeps.

Project: MS3 — Cariaco size-spectrum model.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory


# ---------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------
COLOR_UPTAKE = '#2c7fb8'        # blue: the only source flux on P_i
COLOR_MORT   = '#9e9e9e'        # neutral grey: linear background mortality
COLOR_FISH   = '#d73027'        # red: top-down from outside the size spectrum

# Grazing trio — light to dark = smaller-than-optimal to larger-than-optimal
COLOR_GRAZE_SMALL = '#fdd49e'
COLOR_GRAZE_OPT   = '#f1a340'   # mid orange (the kernel-peak grazer)
COLOR_GRAZE_LARGE = '#7f3b08'

# Pico / Nano / Micro background shading
COLOR_PICO  = '#eef3fb'
COLOR_NANO  = '#fbf6e8'
COLOR_MICRO = '#fbece8'

# Z-source assimilation colours.
# Greens for assimilation from P prey (light to dark = small/opt/large prey).
COLOR_SRC_P_SMALL = '#c7e9c0'
COLOR_SRC_P_OPT   = '#41ab5d'
COLOR_SRC_P_LARGE = '#00441b'
# Blues for assimilation from Z prey (intraguild predation).
COLOR_SRC_Z_SMALL = '#c6dbef'
COLOR_SRC_Z_OPT   = '#4292c6'
COLOR_SRC_Z_LARGE = '#08306b'

# Z size-band shading at 200 µm (net-tow lower limit) and 500 µm (Cariaco bin).
COLOR_BAND_MICROZOO = '#f2f2f2'   # < 200 µm: microzoo, sub-obs
COLOR_BAND_MESO_LO  = '#fbf6e8'   # 200-500 µm: Cariaco (200, 500] bin
COLOR_BAND_MESO_HI  = '#fbece8'   # > 500 µm: Cariaco >500 bin


# ---------------------------------------------------------------
# Predator-group classification
# ---------------------------------------------------------------
def _classify_predator_groups(phiPZ_phyto: np.ndarray):
    """For each phyto class i, classify zoo predators relative to the kernel
    argmax: 0 = smaller-than-optimal, 1 = optimal, 2 = larger-than-optimal.

    The split is defined by the kernel itself (not a fixed ESD bin) so it
    follows theta_opt and sigma_log if they change in a sweep.

    Parameters
    ----------
    phiPZ_phyto : (n_P, n_Z) ndarray
        Phyto-prey rows of the grazing kernel.

    Returns
    -------
    groups : (n_P, n_Z) int ndarray
    j_opt  : (n_P,) int ndarray
    """
    n_P, n_Z = phiPZ_phyto.shape
    j_opt = np.argmax(phiPZ_phyto, axis=1)
    groups = np.full((n_P, n_Z), -1, dtype=int)
    for i in range(n_P):
        groups[i, :j_opt[i]]      = 0   # smaller-than-optimal
        groups[i,  j_opt[i]]      = 1   # optimal
        groups[i,  j_opt[i] + 1:] = 2   # larger-than-optimal
    return groups, j_opt


def _classify_prey_groups(phiPZ_full: np.ndarray, prey_esd: np.ndarray):
    """For each Z predator j, classify each prey k as 0=smaller-than-optimal,
    1=optimal, 2=larger-than-optimal in ESD relative to the kernel argmax.

    Distinct from `_classify_predator_groups`: that function works along the
    predator axis where ESD is index-monotone. Here the prey axis is the
    concatenated [phyto; zoo] grid which is NOT index-monotone in ESD
    (zoo[0]=5 µm sits between phyto[3]=2.6 and phyto[5]=7.6). Comparison
    therefore happens in ESD space rather than index space.

    Parameters
    ----------
    phiPZ_full : (n_prey, n_Z) ndarray
        Full grazing kernel — concatenated [P; Z] prey rows × Z predator cols.
    prey_esd : (n_prey,) ndarray
        Concatenated [phyto_esd, zoo_esd].

    Returns
    -------
    groups : (n_prey, n_Z) int ndarray
        0 = smaller-than-optimal, 1 = optimal, 2 = larger-than-optimal.
    k_opt : (n_Z,) int ndarray
        Argmax prey index per predator.
    """
    k_opt = np.argmax(phiPZ_full, axis=0)
    opt_esd = prey_esd[k_opt]                      # (n_Z,)
    diff = prey_esd[:, None] - opt_esd[None, :]    # (n_prey, n_Z)
    return np.sign(diff).astype(int) + 1, k_opt


# ---------------------------------------------------------------
# Flux extraction from XSO output
# ---------------------------------------------------------------
def extract_phyto_fluxes(out, time_index: int = -1) -> dict:
    """Pull steady-state per-class P fluxes from an XSO output Dataset.

    All flux entries are returned as POSITIVE MAGNITUDES; the sign in the ODE
    is encoded by which list (positive vs negative) the term sits in
    downstream. np.abs() also neutralises floating-point noise sign-flips
    (e.g. -2e-35 instead of +0).

    Parameters
    ----------
    out : xarray.Dataset
        XSO model output containing the variables listed below.
    time_index : int
        Time slice to read (default -1 = last sample = steady-state value
        for the stability solver, which dumps initial + final).

    Returns
    -------
    dict with keys:
        P_ss, Z_ss             — biomass at steady state
        uptake                 — per-class Monod uptake U_i
        mortality              — per-class linear mortality m_P,i * P_i
        fish                   — per-class fish-grazing loss
        grazing_per_predator   — (n_P, n_Z) full G_ij matrix on phyto prey
        phyto_esd, zoo_esd     — size grids
        phiPZ_phyto            — (n_P, n_Z) phyto-prey rows of the kernel
        theta_opt, sigma_log   — kernel parameters used
    """
    sel = dict(time=time_index)

    P     = out['Phytoplankton__biomass'].isel(**sel).values
    Z     = out['Zooplankton__biomass'].isel(**sel).values
    p_esd = out['Phytoplankton__phyto_esd_index'].values
    z_esd = out['Zooplankton__zoo_esd_index'].values
    n_P   = p_esd.size

    U     = out['Growth__uptake_value'].isel(**sel).values
    M     = out['PhytoMortality__mortality_value'].isel(**sel).values
    Ffish = out['FishGrazing__fish_graze_phyto_value'].isel(**sel).values

    # Per-(prey, predator) grazing flux. Take the phyto-prey rows.
    G_full = out['Grazing__grazing_value'].isel(**sel).values   # (full=n_P+n_Z, zoo)
    G_phy  = G_full[:n_P, :]                                    # (n_P, n_Z)

    # Grazing kernel is now serialised directly to the output Dataset.
    # Take the phyto-prey rows (top n_P of the 'full' axis).
    phiPZ_full  = out['Grazing__phiPZ'].values                  # (full, zoo)
    phiPZ_phyto = phiPZ_full[:n_P, :]                           # (n_P, n_Z)
    theta_opt   = float(out['Grazing__theta_opt'].values)
    sigma_log   = float(out['Grazing__sigma_log'].values)

    # Sanity check: per-prey grazing total should match the GGE-side total
    # (sum-over-predators of the matrix vs the GGE component's serialised total).
    if 'GGE__grazing_phyto_value' in out.data_vars:
        graze_total_check = out['GGE__grazing_phyto_value'].isel(**sel).values
        residual = np.max(np.abs(G_phy.sum(axis=1) - graze_total_check))
        if residual > 1e-9:
            print(f'[flux_plot] WARN matrix-vs-GGE grazing residual = {residual:g}')

    # Round FP noise to exact zero. Anything with |value| < EPS is treated
    # as numerical noise from the integrator and rounded down. This stops
    # values like -3.66e-35 or +2.4e-51 from creating phantom near-zero
    # entries that confuse the log-axis autoscaler downstream.
    EPS = 1e-30
    def _denoise(arr):
        a = np.abs(arr)
        return np.where(a < EPS, 0.0, a)

    return {
        'P_ss':                 _denoise(P),
        'Z_ss':                 _denoise(Z),
        'uptake':               _denoise(U),
        'mortality':            _denoise(M),
        'fish':                 _denoise(Ffish),
        'grazing_per_predator': _denoise(G_phy),
        'phyto_esd':            p_esd,
        'zoo_esd':              z_esd,
        'phiPZ_phyto':          phiPZ_phyto,
        'theta_opt':            theta_opt,
        'sigma_log':            sigma_log,
    }


# ---------------------------------------------------------------
# Quantitative summary
# ---------------------------------------------------------------
def summarize_phyto_fluxes(
    out,
    *,
    time_index: int = -1,
    extinct_threshold: float = 1e-7,
    return_dict: bool = True,
):
    """Print a quantitative per-class P flux breakdown and return the data.

    Three sections printed:
      1. Absolute fluxes per class [mmol N m-3 d-1], with each loss term's
         fraction of that class's total loss.
      2. Per-capita rates [d-1] (= absolute flux / P_ss). At steady state the
         net per-capita rate is ~0 for surviving classes; the magnitude of
         the gross rate is a useful intensive measure for comparison across
         classes and against the analytical R*-style derivation.
      3. Dominant loss term per surviving class — quickly identifies what
         controls each class's steady-state biomass.

    Parameters
    ----------
    out : xarray.Dataset
        XSO model output (steady-state run).
    time_index : int, default -1
    extinct_threshold : float
        Classes with P_ss below this are flagged '<extinct>' and skipped
        in the per-capita table.
    return_dict : bool, default True
        If True (default), return a dict of per-class arrays for further
        programmatic use.

    Returns
    -------
    dict or None
    """
    f = extract_phyto_fluxes(out, time_index=time_index)
    P_ss  = f['P_ss']
    p_esd = f['phyto_esd']
    n_P   = P_ss.size

    U  = f['uptake']
    M  = f['mortality']
    Ff = f['fish']
    G  = f['grazing_per_predator']

    groups, _ = _classify_predator_groups(f['phiPZ_phyto'])
    G_small = np.where(groups == 0, G, 0.0).sum(axis=1)
    G_opt   = np.where(groups == 1, G, 0.0).sum(axis=1)
    G_large = np.where(groups == 2, G, 0.0).sum(axis=1)
    total_loss = G_small + G_opt + G_large + M + Ff
    net = U - total_loss

    alive = np.isfinite(P_ss) & (P_ss > extinct_threshold)

    # Header
    print('=' * 116)
    print('Per-class phytoplankton flux decomposition — steady state')
    print('=' * 116)
    try:
        fn  = float(out['Inflow__FN'].values)
        de  = float(out['Inflow__de'].values)
        ksz = float(out['Grazing__KsZ'].values)
        gge = float(out['GGE__gge'].values)
        print(f'  Run params: F_N={fn:.4g}, d_e={de:.4g}, K_sZ={ksz:.4g}, GGE={gge:.4g}')
    except KeyError:
        pass
    print('  Units: mmol N m^-3 d^-1; (xx%) = fraction of total loss for that class')
    print()

    # --- Section 1: Absolute fluxes ---
    hdr = (f"{'i':>2}  {'ESD':>7}  {'P_ss':>10}  | "
           f"{'Uptake':>10}  || "
           f"{'Gr<opt':>15}  {'Gr=opt':>15}  {'Gr>opt':>15}  "
           f"{'Mort.':>15}  {'Fish':>15}  | "
           f"{'Σloss':>10}  {'Net':>10}")
    print(hdr)
    print('-' * len(hdr))

    def _fmt_loss(x, tot):
        if tot > 0:
            return f"{x:>8.2e} ({100*x/tot:>3.0f}%)"
        return f"{x:>8.2e} (  -)"

    for i in range(n_P):
        esd_s = f'{p_esd[i]:.3g}'
        if not alive[i]:
            print(f"{i:>2}  {esd_s:>7}  {'<extinct>':>10}")
            continue
        tot = total_loss[i]
        print(f"{i:>2}  {esd_s:>7}  {P_ss[i]:>10.3e}  | "
              f"{U[i]:>+10.3e}  || "
              f"{_fmt_loss(G_small[i], tot)}  "
              f"{_fmt_loss(G_opt[i],   tot)}  "
              f"{_fmt_loss(G_large[i], tot)}  "
              f"{_fmt_loss(M[i],       tot)}  "
              f"{_fmt_loss(Ff[i],      tot)}  | "
              f"{tot:>10.3e}  {net[i]:>+10.2e}")

    # --- Section 2: Per-capita rates ---
    print()
    print('Per-capita rates [d^-1]  (flux / P_ss):')
    hdr2 = (f"{'i':>2}  {'ESD':>7}  | "
            f"{'µ_uptake':>10}  ||  "
            f"{'r_Gr<opt':>10}  {'r_Gr=opt':>10}  {'r_Gr>opt':>10}  "
            f"{'r_mort':>10}  {'r_fish':>10}  | "
            f"{'Σloss':>10}")
    print(hdr2)
    print('-' * len(hdr2))
    for i in range(n_P):
        esd_s = f'{p_esd[i]:.3g}'
        if not alive[i]:
            continue
        invP = 1.0 / P_ss[i]
        print(f"{i:>2}  {esd_s:>7}  | "
              f"{U[i]*invP:>+10.3e}  ||  "
              f"{G_small[i]*invP:>10.3e}  "
              f"{G_opt[i]*invP:>10.3e}  "
              f"{G_large[i]*invP:>10.3e}  "
              f"{M[i]*invP:>10.3e}  "
              f"{Ff[i]*invP:>10.3e}  | "
              f"{total_loss[i]*invP:>10.3e}")

    # --- Section 3: Dominant loss per class ---
    print()
    print('Dominant loss term per surviving class:')
    loss_names = ['Gr<opt', 'Gr=opt', 'Gr>opt', 'Mort.', 'Fish']
    loss_arrs  = np.array([G_small, G_opt, G_large, M, Ff])  # (5, n_P)
    for i in range(n_P):
        if not alive[i]:
            continue
        tot = total_loss[i] if total_loss[i] > 0 else 1.0
        idx = int(np.argmax(loss_arrs[:, i]))
        pct = 100 * loss_arrs[idx, i] / tot
        print(f"  i={i:>2}  ESD={p_esd[i]:>7.3g} µm   →  "
              f"{loss_names[idx]:<7}  ({pct:>4.0f}% of total loss)")
    print('=' * 116)

    if return_dict:
        return {
            'phyto_esd':   p_esd,
            'P_ss':        P_ss,
            'uptake':      U,
            'graze_small': G_small,
            'graze_opt':   G_opt,
            'graze_large': G_large,
            'mortality':   M,
            'fish':        Ff,
            'total_loss':  total_loss,
            'net_rate':    net,
            'alive':       alive,
        }
    return None


# ---------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------
def plot_phyto_flux_decomposition(
    out,
    *,
    time_index: int = -1,
    mode: str = 'biomass',
    dt: float = 1.0,
    extinct_threshold: float = 1e-7,
    bar_width: float = 0.7,
    show_size_bands: bool = True,
    ax=None,
    title: str | None = None,
):
    """Per-class phyto flux decomposition with steady-state biomass at the centre.

    Parameters
    ----------
    out : xarray.Dataset
        XSO steady-state model output.
    time_index : int, default -1
        Time index to read; -1 = last step = steady-state value for the
        stability solver (which dumps initial + final).
    mode : {'biomass', 'per_capita'}, default 'biomass'
        - 'biomass'    : log-y biomass axis; bars centred at P_i*; segment
                         lengths are flux * dt (mass-equivalent over `dt` d).
                         Lets you see the size spectrum as the centre line.
        - 'per_capita' : linear-y per-capita rate axis; bars centred at 0;
                         segment lengths are per-capita rates [d-1].
                         Matches the analytical-derivation framing.
    dt : float, default 1.0
        Time scale in days for flux→mass conversion in 'biomass' mode.
        Decrease (e.g. 0.1) if very-fast-turnover classes push the bar bottom
        below zero on the log axis (segments are clipped to a small floor).
    extinct_threshold : float
        Phyto classes with P_i* below this are masked from the bars and
        marked 'ext' on the x-axis.
    bar_width : float
        Bar width in fraction of one x-unit.
    show_size_bands : bool
        Background shading for Pico (<2 µm), Nano (2–20 µm), Micro (>20 µm).
    ax : matplotlib Axes or None
        If None, a new figure+axes are created.
    title : str or None
        Plot title; default is auto-generated from `mode` and `dt`.

    Returns
    -------
    ax : matplotlib Axes
    """
    f = extract_phyto_fluxes(out, time_index=time_index)
    P_ss = f['P_ss']
    p_esd = f['phyto_esd']
    n_P = P_ss.size

    U  = f['uptake']
    M  = f['mortality']
    Ff = f['fish']
    G  = f['grazing_per_predator']

    groups, j_opt = _classify_predator_groups(f['phiPZ_phyto'])
    G_small = np.where(groups == 0, G, 0.0).sum(axis=1)
    G_opt   = np.where(groups == 1, G, 0.0).sum(axis=1)
    G_large = np.where(groups == 2, G, 0.0).sum(axis=1)

    # NaN-safe alive mask: NaN > x evaluates to False, so NaN classes are out.
    alive = np.isfinite(P_ss) & (P_ss > extinct_threshold)

    if not alive.any():
        n_nan = int((~np.isfinite(P_ss)).sum())
        n_below = int(((P_ss <= extinct_threshold) & np.isfinite(P_ss)).sum())
        raise RuntimeError(
            f"plot_phyto_flux_decomposition: no phyto classes alive at "
            f"time_index={time_index} (P_ss > {extinct_threshold}). "
            f"{n_nan}/{n_P} entries are NaN/inf, {n_below}/{n_P} at/below "
            f"threshold. Check "
            f"`out['Phytoplankton__biomass'].isel(time={time_index}).values` "
            f"— if it is NaN the upstream model run failed silently."
        )

    # ---- mode-specific quantities ----
    if mode == 'per_capita':
        with np.errstate(divide='ignore', invalid='ignore'):
            scale = np.where(alive, 1.0 / P_ss, 0.0)
        centre = np.zeros(n_P)
        positive_terms = [('Uptake', U * scale, COLOR_UPTAKE)]
        negative_terms = [
            ('Graze (Z<opt)', G_small * scale, COLOR_GRAZE_SMALL),
            ('Graze (Z=opt)', G_opt   * scale, COLOR_GRAZE_OPT),
            ('Graze (Z>opt)', G_large * scale, COLOR_GRAZE_LARGE),
            ('Linear mort.',  M  * scale,      COLOR_MORT),
            ('Fish grazing',  Ff * scale,      COLOR_FISH),
        ]
        y_label = 'Per-capita flux (d$^{-1}$)'
        log_y = False
    elif mode == 'biomass':
        centre = P_ss.copy()
        positive_terms = [('Uptake', U * dt, COLOR_UPTAKE)]
        negative_terms = [
            ('Graze (Z<opt)', G_small * dt, COLOR_GRAZE_SMALL),
            ('Graze (Z=opt)', G_opt   * dt, COLOR_GRAZE_OPT),
            ('Graze (Z>opt)', G_large * dt, COLOR_GRAZE_LARGE),
            ('Linear mort.',  M  * dt,      COLOR_MORT),
            ('Fish grazing',  Ff * dt,      COLOR_FISH),
        ]
        y_label = (f'Biomass [mmol N m$^{{-3}}$]   '
                   f'(±flux × {dt:g} d either side of centre)')
        log_y = True
    else:
        raise ValueError(f"mode must be 'biomass' or 'per_capita', got {mode!r}")

    # ---- canvas ----
    if ax is None:
        fig, ax = plt.subplots(figsize=(11.5, 6.5))
    else:
        fig = ax.figure
    x = np.arange(n_P)

    # ---- size-band background ----
    if show_size_bands:
        # Pico/Nano/Micro boundaries at 2 and 20 µm; interpolate to x-coords.
        def esd_to_x(esd):
            return np.interp(np.log10(esd), np.log10(p_esd), x)
        x_pn = esd_to_x(2.0)
        x_nm = esd_to_x(20.0)
        x_lo, x_hi = -0.5, n_P - 0.5
        ax.axvspan(x_lo, x_pn, color=COLOR_PICO,  alpha=0.7, zorder=0)
        ax.axvspan(x_pn, x_nm, color=COLOR_NANO,  alpha=0.7, zorder=0)
        ax.axvspan(x_nm, x_hi, color=COLOR_MICRO, alpha=0.7, zorder=0)
        # Region labels at the top
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        ax.text((x_lo + x_pn) / 2, 0.97, 'Pico',  transform=trans,
                ha='center', va='top', fontsize=9, color='dimgrey', alpha=0.8)
        ax.text((x_pn + x_nm) / 2, 0.97, 'Nano',  transform=trans,
                ha='center', va='top', fontsize=9, color='dimgrey', alpha=0.8)
        ax.text((x_nm + x_hi) / 2, 0.97, 'Micro', transform=trans,
                ha='center', va='top', fontsize=9, color='dimgrey', alpha=0.8)

    # Draw bars ONLY for alive classes — never pass NaN or zero-height bars
    # to matplotlib, which avoids polluting the log-axis data range with
    # phantom near-zero data points.
    alive_idx = np.where(alive)[0]
    if alive_idx.size > 0:
        x_a = alive_idx.astype(float)
        c_a = centre[alive_idx]

        # ---- positive (source) segments stacked above centre ----
        bottom = c_a.copy()
        for label, values, color in positive_terms:
            v_a = values[alive_idx]
            ax.bar(x_a, v_a, bottom=bottom, width=bar_width,
                   color=color, edgecolor='black', linewidth=0.3,
                   label=label, zorder=3)
            bottom = bottom + v_a

        # ---- negative (sink) segments stacked below centre ----
        top = c_a.copy()
        floor_a = 1e-3 * P_ss[alive_idx] if mode == 'biomass' else None
        for label, values, color in negative_terms:
            v_a = values[alive_idx]
            if mode == 'biomass':
                # Don't let stacked sinks push the bar bottom below
                # 1e-3 * P_ss (which would crash the log axis).
                v_clip = np.minimum(v_a, np.maximum(top - floor_a, 0.0))
            else:
                v_clip = v_a
            ax.bar(x_a, -v_clip, bottom=top, width=bar_width,
                   color=color, edgecolor='black', linewidth=0.3,
                   label=label, zorder=3)
            top = top - v_clip

        # ---- centre tick (P_ss line) for alive classes ----
        for xi, c in zip(x_a, c_a):
            ax.hlines(c, xi - bar_width / 2, xi + bar_width / 2,
                      colors='black', linewidth=1.4, zorder=5)

    # ---- extinct-class markers (axes-fraction y, mid-axis vertically) ----
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for xi in x[~alive]:
        ax.text(xi, 0.5, 'ext', transform=trans,
                ha='center', va='center', fontsize=8, color='dimgrey',
                style='italic',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor='lightgrey', alpha=0.85))

    # ---- axes ----
    ax.set_xticks(x)
    ax.set_xticklabels([f'{e:.2g}' for e in p_esd], rotation=45, ha='right')
    ax.set_xlabel('Phyto ESD (µm)')
    ax.set_ylabel(y_label)
    ax.set_xlim(-0.5, n_P - 0.5)
    if log_y:
        # Tight log-y limits: compute the actual top/bottom of the drawn bars
        # and pad ~0.3 decades (factor ~2) on each side. Honour the floor
        # clipping used in the negative-segment loop above so bar_bot can't
        # drop below 1e-3 * P_ss.
        if alive_idx.size > 0:
            P_alive = P_ss[alive_idx]
            sum_pos = sum(v[alive_idx] for _, v, _ in positive_terms)
            sum_neg = sum(v[alive_idx] for _, v, _ in negative_terms)
            bar_top = P_alive + sum_pos
            bar_bot = np.maximum(P_alive - sum_neg, 1e-3 * P_alive)
            ymin = max(bar_bot.min() * 0.5, 1e-30)
            ymax = bar_top.max() * 2.0
        else:
            ymin, ymax = 1e-3, 1.0
        ax.set_ylim(ymin, ymax)
        ax.set_yscale('log')

    if title is None:
        title = ('Per-class P flux decomposition — '
                 + ('per-capita rates' if mode == 'per_capita'
                    else f'biomass with ±flux × {dt:g} d'))
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8, ncol=2, framealpha=0.95)
    ax.grid(axis='y', linestyle=':', alpha=0.4, zorder=1)

    return ax


# =================================================================
# ZOOPLANKTON
# =================================================================
#
# Per-class Z source/sink terms:
#   Source:  assimilation = gge * I_j  where I_j = Σ_k G_kj summed across
#            all 24 prey items (P + Z). The source is decomposed by prey
#            type (P / Z) AND by prey-size group (small / opt / large)
#            relative to each predator's kernel argmax → 6 segments above
#            the centre. P shown in greens, Z in blues.
#   Sinks:   per-class grazing on Z (Z as prey), decomposed into 3 predator
#            groups (small / opt / large) using the same scheme as the P
#            plot; quadratic closure m_Z * Z_j * ΣZ; fish grazing on Z.
#
# Loss-side colours (greens, oranges, brown, grey, red) are reused from
# the P palette so the two figures read consistently in the manuscript.


def extract_zoo_fluxes(out, time_index: int = -1) -> dict:
    """Pull steady-state per-class Z fluxes from an XSO output Dataset.

    Returns a dict carrying the per-(prey, predator) grazing matrix, the
    per-class closure and fish loss, the kernel, and steady-state biomass.
    The source-side decomposition is computed downstream from the matrix
    so that the same extraction can serve both `plot_zoo_flux_decomposition`
    and `summarize_zoo_fluxes`.

    Returns
    -------
    dict with keys:
        P_ss, Z_ss     — steady-state biomass
        gge            — scalar gross growth efficiency
        G_full         — (n_P+n_Z, n_Z) per-(prey, predator) ingestion flux
        closure        — (n_Z,) per-class quadratic mortality
        fish           — (n_Z,) per-class fish-grazing loss
        phiPZ_full     — (n_P+n_Z, n_Z) full grazing kernel
        prey_esd       — (n_P+n_Z,) concatenated [phyto_esd, zoo_esd]
        phyto_esd, zoo_esd, theta_opt, sigma_log
    """
    sel = dict(time=time_index)

    P     = out['Phytoplankton__biomass'].isel(**sel).values
    Z     = out['Zooplankton__biomass'].isel(**sel).values
    p_esd = out['Phytoplankton__phyto_esd_index'].values
    z_esd = out['Zooplankton__zoo_esd_index'].values

    gge   = float(out['GGE__gge'].values)

    # Per-(prey, predator) grazing flux. Phyto rows 0..n_P-1, zoo rows n_P:.
    G_full = out['Grazing__grazing_value'].isel(**sel).values

    # Per-class Z closure (quadratic mortality) and fish loss.
    M_z   = out['ZooMortality__mortality_value'].isel(**sel).values
    Ff_z  = out['FishGrazing__fish_graze_zoo_value'].isel(**sel).values

    # Full kernel from the framework (post-XSO update).
    phiPZ_full = out['Grazing__phiPZ'].values

    EPS = 1e-30
    def _denoise(arr):
        a = np.abs(arr)
        return np.where(a < EPS, 0.0, a)

    return {
        'P_ss':       _denoise(P),
        'Z_ss':       _denoise(Z),
        'gge':        gge,
        'G_full':     _denoise(G_full),
        'closure':    _denoise(M_z),
        'fish':       _denoise(Ff_z),
        'phiPZ_full': phiPZ_full,
        'prey_esd':   np.concatenate([p_esd, z_esd]),
        'phyto_esd':  p_esd,
        'zoo_esd':    z_esd,
        'theta_opt':  float(out['Grazing__theta_opt'].values),
        'sigma_log':  float(out['Grazing__sigma_log'].values),
    }


def _decompose_zoo_fluxes(f: dict):
    """Compute the 6 source segments and 3 grazing-loss segments for Z.

    Used internally by both the plot and summary functions to keep them
    consistent.
    """
    p_esd = f['phyto_esd']
    z_esd = f['zoo_esd']
    n_P, n_Z = p_esd.size, z_esd.size
    gge = f['gge']
    G_full = f['G_full']
    prey_esd = f['prey_esd']
    is_phyto = np.arange(n_P + n_Z) < n_P  # bool mask along prey axis

    # SOURCE: prey-side decomposition
    prey_groups, k_opt = _classify_prey_groups(f['phiPZ_full'], prey_esd)

    def _src(prey_filter, gid):
        mask = (prey_groups == gid) & prey_filter[:, None]
        return (gge * G_full * mask).sum(axis=0)  # (n_Z,)

    src = {
        'P_small': _src(is_phyto,  0),
        'P_opt':   _src(is_phyto,  1),
        'P_large': _src(is_phyto,  2),
        'Z_small': _src(~is_phyto, 0),
        'Z_opt':   _src(~is_phyto, 1),
        'Z_large': _src(~is_phyto, 2),
    }
    src['from_P'] = src['P_small'] + src['P_opt'] + src['P_large']
    src['from_Z'] = src['Z_small'] + src['Z_opt'] + src['Z_large']
    src['total']  = src['from_P'] + src['from_Z']

    # SINK: grazing on Z (Z as prey), classified by predator size relative
    # to each Z-prey class's kernel-argmax predator.
    G_on_Z = G_full[n_P:, :]                                    # (n_Z prey, n_Z pred)
    pred_groups, j_opt = _classify_predator_groups(f['phiPZ_full'][n_P:, :])
    grz = {
        'pred_small': np.where(pred_groups == 0, G_on_Z, 0.0).sum(axis=1),
        'pred_opt':   np.where(pred_groups == 1, G_on_Z, 0.0).sum(axis=1),
        'pred_large': np.where(pred_groups == 2, G_on_Z, 0.0).sum(axis=1),
    }
    grz['total_graze'] = grz['pred_small'] + grz['pred_opt'] + grz['pred_large']

    return src, grz, k_opt, j_opt


# ---------------------------------------------------------------
# Z quantitative summary
# ---------------------------------------------------------------
def summarize_zoo_fluxes(
    out,
    *,
    time_index: int = -1,
    extinct_threshold: float = 1e-7,
    return_dict: bool = True,
):
    """Print a quantitative per-class Z flux breakdown and return the data.

    Four sections printed:
      1. Source side: assimilation per class decomposed into 6 segments
         (P<opt / P=opt / P>opt / Z<opt / Z=opt / Z>opt). Each cell shows
         the absolute flux [mmol N m-3 d-1] and its fraction of total source.
      2. Sink side: grazing-on-Z (3 predator groups) + closure + fish.
      3. Per-capita rates [d-1] for source and each loss term.
      4. Dominant source AND dominant loss term per surviving class, plus
         a herbivory/carnivory tag based on the P-vs-Z source split.
    """
    f = extract_zoo_fluxes(out, time_index=time_index)
    Z_ss  = f['Z_ss']
    z_esd = f['zoo_esd']
    n_Z   = Z_ss.size
    gge   = f['gge']

    src, grz, _, _ = _decompose_zoo_fluxes(f)
    M  = f['closure']
    Ff = f['fish']
    total_loss = grz['total_graze'] + M + Ff
    net = src['total'] - total_loss

    alive = np.isfinite(Z_ss) & (Z_ss > extinct_threshold)

    print('=' * 142)
    print('Per-class zooplankton flux decomposition — steady state')
    print('=' * 142)
    try:
        ksz = float(out['Grazing__KsZ'].values)
        mz  = float(out['ZooMortality__rate'].values)
        fr  = float(out['FishGrazing__rate'].values)
        print(f'  Run params: gge={gge:.4g}, K_sZ={ksz:.4g}, '
              f'm_Z={mz:.4g}, fish_rate={fr:.4g}')
    except KeyError:
        pass
    print('  Units: mmol N m^-3 d^-1; (xx%) = fraction of total source/loss for that class')
    print()

    def _fmt(x, tot):
        if tot > 0:
            return f"{x:>8.2e} ({100*x/tot:>3.0f}%)"
        return f"{x:>8.2e} (  -)"

    # --- Section 1: Sources ---
    print('SOURCE SIDE — gge x ingestion, by prey type and prey-size relative to optimal')
    hdr = (f"{'j':>2}  {'ESD':>8}  {'Z_ss':>10}  | "
           f"{'P<opt':>14}  {'P=opt':>14}  {'P>opt':>14}  ||  "
           f"{'Z<opt':>14}  {'Z=opt':>14}  {'Z>opt':>14}  | "
           f"{'Σ src':>10}  {'P:Z share':>12}")
    print(hdr)
    print('-' * len(hdr))
    for j in range(n_Z):
        esd_s = f'{z_esd[j]:.4g}'
        if not alive[j]:
            print(f"{j:>2}  {esd_s:>8}  {'<extinct>':>10}")
            continue
        tot = src['total'][j]
        if tot > 0:
            pz_share = f"{100*src['from_P'][j]/tot:>3.0f}% / {100*src['from_Z'][j]/tot:>3.0f}%"
        else:
            pz_share = '   -  /   -'
        print(f"{j:>2}  {esd_s:>8}  {Z_ss[j]:>10.3e}  | "
              f"{_fmt(src['P_small'][j], tot)}  "
              f"{_fmt(src['P_opt'][j],   tot)}  "
              f"{_fmt(src['P_large'][j], tot)}  ||  "
              f"{_fmt(src['Z_small'][j], tot)}  "
              f"{_fmt(src['Z_opt'][j],   tot)}  "
              f"{_fmt(src['Z_large'][j], tot)}  | "
              f"{tot:>10.3e}  {pz_share:>12}")

    # --- Section 2: Sinks ---
    print()
    print('SINK SIDE — predation on Z (3 predator groups), closure, fish')
    hdr2 = (f"{'j':>2}  {'ESD':>8}  | "
            f"{'Pr<opt':>14}  {'Pr=opt':>14}  {'Pr>opt':>14}  | "
            f"{'Closure':>14}  {'Fish':>14}  | "
            f"{'Σ loss':>10}  {'Net':>11}")
    print(hdr2)
    print('-' * len(hdr2))
    for j in range(n_Z):
        if not alive[j]:
            continue
        tot = total_loss[j]
        print(f"{j:>2}  {z_esd[j]:>8.4g}  | "
              f"{_fmt(grz['pred_small'][j], tot)}  "
              f"{_fmt(grz['pred_opt'][j],   tot)}  "
              f"{_fmt(grz['pred_large'][j], tot)}  | "
              f"{_fmt(M[j],                  tot)}  "
              f"{_fmt(Ff[j],                 tot)}  | "
              f"{tot:>10.3e}  {net[j]:>+11.3e}")

    # --- Section 3: Per-capita rates ---
    print()
    print('Per-capita rates [d^-1]:')
    hdr3 = (f"{'j':>2}  {'ESD':>8}  | "
            f"{'r_src':>10}  ||  "
            f"{'r_pred':>10}  {'r_clos':>10}  {'r_fish':>10}  | "
            f"{'Σloss':>10}")
    print(hdr3)
    print('-' * len(hdr3))
    for j in range(n_Z):
        if not alive[j]:
            continue
        invZ = 1.0 / Z_ss[j]
        print(f"{j:>2}  {z_esd[j]:>8.4g}  | "
              f"{src['total'][j]*invZ:>+10.3e}  ||  "
              f"{grz['total_graze'][j]*invZ:>10.3e}  "
              f"{M[j]*invZ:>10.3e}  "
              f"{Ff[j]*invZ:>10.3e}  | "
              f"{total_loss[j]*invZ:>10.3e}")

    # --- Section 4: Dominant terms ---
    print()
    print('Dominant SOURCE term per surviving class:')
    src_names = ['P<opt', 'P=opt', 'P>opt', 'Z<opt', 'Z=opt', 'Z>opt']
    src_arrs  = np.array([src['P_small'], src['P_opt'], src['P_large'],
                          src['Z_small'], src['Z_opt'], src['Z_large']])
    for j in range(n_Z):
        if not alive[j]:
            continue
        tot = src['total'][j] if src['total'][j] > 0 else 1.0
        idx = int(np.argmax(src_arrs[:, j]))
        pct = 100 * src_arrs[idx, j] / tot
        carn_pct = 100 * src['from_Z'][j] / tot
        if   carn_pct < 10:  trophic = 'herbivore'
        elif carn_pct > 90:  trophic = 'carnivore'
        else:                trophic = 'omnivore '
        print(f"  j={j:>2}  ESD={z_esd[j]:>8.4g} µm   →  "
              f"{src_names[idx]:<6}  ({pct:>4.0f}% src)  "
              f"[{trophic}: {100-carn_pct:>3.0f}% P / {carn_pct:>3.0f}% Z]")

    print()
    print('Dominant LOSS term per surviving class:')
    loss_names = ['Pr<opt', 'Pr=opt', 'Pr>opt', 'Closure', 'Fish']
    loss_arrs  = np.array([grz['pred_small'], grz['pred_opt'], grz['pred_large'],
                           M, Ff])
    for j in range(n_Z):
        if not alive[j]:
            continue
        tot = total_loss[j] if total_loss[j] > 0 else 1.0
        idx = int(np.argmax(loss_arrs[:, j]))
        pct = 100 * loss_arrs[idx, j] / tot
        print(f"  j={j:>2}  ESD={z_esd[j]:>8.4g} µm   →  "
              f"{loss_names[idx]:<7}  ({pct:>4.0f}% of total loss)")
    print('=' * 142)

    if return_dict:
        return {
            'zoo_esd':          z_esd,
            'Z_ss':             Z_ss,
            'src_P_small':      src['P_small'],
            'src_P_opt':        src['P_opt'],
            'src_P_large':      src['P_large'],
            'src_Z_small':      src['Z_small'],
            'src_Z_opt':        src['Z_opt'],
            'src_Z_large':      src['Z_large'],
            'src_from_P':       src['from_P'],
            'src_from_Z':       src['from_Z'],
            'total_src':        src['total'],
            'graze_pred_small': grz['pred_small'],
            'graze_pred_opt':   grz['pred_opt'],
            'graze_pred_large': grz['pred_large'],
            'closure':          M,
            'fish':             Ff,
            'total_loss':       total_loss,
            'net_rate':         net,
            'alive':            alive,
        }
    return None


# ---------------------------------------------------------------
# Z plotting function
# ---------------------------------------------------------------
def plot_zoo_flux_decomposition(
    out,
    *,
    time_index: int = -1,
    mode: str = 'biomass',
    dt: float = 1.0,
    extinct_threshold: float = 1e-7,
    bar_width: float = 0.7,
    show_size_bands: bool = True,
    ax=None,
    title: str | None = None,
):
    """Per-class zooplankton flux decomposition with steady-state biomass at the centre.

    Parameters mirror `plot_phyto_flux_decomposition`. Source side has 6
    stacked segments above the centre (P-prey assimilation in greens; Z-prey
    assimilation in blues; light/mid/dark = small/opt/large prey relative to
    that predator's kernel argmax). Sink side reuses the P-plot palette:
    grazing-on-Z by predator group (oranges/brown), closure (grey), fish (red).
    Background bands at 200 and 500 µm correspond to the Cariaco zoo obs bins.
    """
    f = extract_zoo_fluxes(out, time_index=time_index)
    Z_ss  = f['Z_ss']
    z_esd = f['zoo_esd']
    n_Z   = Z_ss.size

    src, grz, _, _ = _decompose_zoo_fluxes(f)
    M  = f['closure']
    Ff = f['fish']

    alive = np.isfinite(Z_ss) & (Z_ss > extinct_threshold)
    if not alive.any():
        n_nan = int((~np.isfinite(Z_ss)).sum())
        n_below = int(((Z_ss <= extinct_threshold) & np.isfinite(Z_ss)).sum())
        raise RuntimeError(
            f"plot_zoo_flux_decomposition: no zoo classes alive at "
            f"time_index={time_index} (Z_ss > {extinct_threshold}). "
            f"{n_nan}/{n_Z} entries are NaN/inf, {n_below}/{n_Z} at/below "
            f"threshold. Check `out['Zooplankton__biomass'].isel(time={time_index}).values`."
        )

    # ---- mode-specific quantities ----
    if mode == 'per_capita':
        with np.errstate(divide='ignore', invalid='ignore'):
            scale = np.where(alive, 1.0 / Z_ss, 0.0)
        centre = np.zeros(n_Z)
        positive_terms = [
            ('Assim P<opt', src['P_small'] * scale, COLOR_SRC_P_SMALL),
            ('Assim P=opt', src['P_opt']   * scale, COLOR_SRC_P_OPT),
            ('Assim P>opt', src['P_large'] * scale, COLOR_SRC_P_LARGE),
            ('Assim Z<opt', src['Z_small'] * scale, COLOR_SRC_Z_SMALL),
            ('Assim Z=opt', src['Z_opt']   * scale, COLOR_SRC_Z_OPT),
            ('Assim Z>opt', src['Z_large'] * scale, COLOR_SRC_Z_LARGE),
        ]
        negative_terms = [
            ('Pred Z<opt', grz['pred_small'] * scale, COLOR_GRAZE_SMALL),
            ('Pred Z=opt', grz['pred_opt']   * scale, COLOR_GRAZE_OPT),
            ('Pred Z>opt', grz['pred_large'] * scale, COLOR_GRAZE_LARGE),
            ('Closure',    M  * scale, COLOR_MORT),
            ('Fish',       Ff * scale, COLOR_FISH),
        ]
        y_label = 'Per-capita flux (d$^{-1}$)'
        log_y = False
    elif mode == 'biomass':
        centre = Z_ss.copy()
        positive_terms = [
            ('Assim P<opt', src['P_small'] * dt, COLOR_SRC_P_SMALL),
            ('Assim P=opt', src['P_opt']   * dt, COLOR_SRC_P_OPT),
            ('Assim P>opt', src['P_large'] * dt, COLOR_SRC_P_LARGE),
            ('Assim Z<opt', src['Z_small'] * dt, COLOR_SRC_Z_SMALL),
            ('Assim Z=opt', src['Z_opt']   * dt, COLOR_SRC_Z_OPT),
            ('Assim Z>opt', src['Z_large'] * dt, COLOR_SRC_Z_LARGE),
        ]
        negative_terms = [
            ('Pred Z<opt', grz['pred_small'] * dt, COLOR_GRAZE_SMALL),
            ('Pred Z=opt', grz['pred_opt']   * dt, COLOR_GRAZE_OPT),
            ('Pred Z>opt', grz['pred_large'] * dt, COLOR_GRAZE_LARGE),
            ('Closure',    M  * dt, COLOR_MORT),
            ('Fish',       Ff * dt, COLOR_FISH),
        ]
        y_label = (f'Biomass [mmol N m$^{{-3}}$]   '
                   f'(±flux × {dt:g} d either side of centre)')
        log_y = True
    else:
        raise ValueError(f"mode must be 'biomass' or 'per_capita', got {mode!r}")

    # ---- canvas ----
    if ax is None:
        fig, ax = plt.subplots(figsize=(12.5, 7.0))
    else:
        fig = ax.figure
    x = np.arange(n_Z)

    # ---- size bands at 200 and 500 µm ----
    if show_size_bands:
        def esd_to_x(esd):
            return np.interp(np.log10(esd), np.log10(z_esd), x)
        x_200 = esd_to_x(200.0)
        x_500 = esd_to_x(500.0)
        x_lo, x_hi = -0.5, n_Z - 0.5
        ax.axvspan(x_lo,  x_200, color=COLOR_BAND_MICROZOO, alpha=0.7, zorder=0)
        ax.axvspan(x_200, x_500, color=COLOR_BAND_MESO_LO,  alpha=0.7, zorder=0)
        ax.axvspan(x_500, x_hi,  color=COLOR_BAND_MESO_HI,  alpha=0.7, zorder=0)
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        ax.text((x_lo  + x_200) / 2, 0.97, 'Microzoo (sub-obs)',
                transform=trans, ha='center', va='top',
                fontsize=9, color='dimgrey', alpha=0.8)
        ax.text((x_200 + x_500) / 2, 0.97, 'Zoo (200, 500]',
                transform=trans, ha='center', va='top',
                fontsize=9, color='dimgrey', alpha=0.8)
        ax.text((x_500 + x_hi)  / 2, 0.97, 'Zoo > 500',
                transform=trans, ha='center', va='top',
                fontsize=9, color='dimgrey', alpha=0.8)

    # ---- bars (alive classes only) ----
    alive_idx = np.where(alive)[0]
    if alive_idx.size > 0:
        x_a = alive_idx.astype(float)
        c_a = centre[alive_idx]

        bottom = c_a.copy()
        for label, values, color in positive_terms:
            v_a = values[alive_idx]
            ax.bar(x_a, v_a, bottom=bottom, width=bar_width,
                   color=color, edgecolor='black', linewidth=0.3,
                   label=label, zorder=3)
            bottom = bottom + v_a

        top = c_a.copy()
        floor_a = 1e-3 * Z_ss[alive_idx] if mode == 'biomass' else None
        for label, values, color in negative_terms:
            v_a = values[alive_idx]
            if mode == 'biomass':
                v_clip = np.minimum(v_a, np.maximum(top - floor_a, 0.0))
            else:
                v_clip = v_a
            ax.bar(x_a, -v_clip, bottom=top, width=bar_width,
                   color=color, edgecolor='black', linewidth=0.3,
                   label=label, zorder=3)
            top = top - v_clip

        for xi, c in zip(x_a, c_a):
            ax.hlines(c, xi - bar_width / 2, xi + bar_width / 2,
                      colors='black', linewidth=1.4, zorder=5)

    # ---- extinct markers ----
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for xi in x[~alive]:
        ax.text(xi, 0.5, 'ext', transform=trans,
                ha='center', va='center', fontsize=8, color='dimgrey',
                style='italic',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor='lightgrey', alpha=0.85))

    # ---- axes ----
    ax.set_xticks(x)
    ax.set_xticklabels([f'{e:.4g}' for e in z_esd], rotation=45, ha='right')
    ax.set_xlabel('Zoo ESD (µm)')
    ax.set_ylabel(y_label)
    ax.set_xlim(-0.5, n_Z - 0.5)
    if log_y:
        if alive_idx.size > 0:
            Z_alive = Z_ss[alive_idx]
            sum_pos = sum(v[alive_idx] for _, v, _ in positive_terms)
            sum_neg = sum(v[alive_idx] for _, v, _ in negative_terms)
            bar_top = Z_alive + sum_pos
            bar_bot = np.maximum(Z_alive - sum_neg, 1e-3 * Z_alive)
            ymin = max(bar_bot.min() * 0.5, 1e-30)
            ymax = bar_top.max() * 2.0
        else:
            ymin, ymax = 1e-3, 1.0
        ax.set_ylim(ymin, ymax)
        ax.set_yscale('log')

    if title is None:
        title = ('Per-class Z flux decomposition — '
                 + ('per-capita rates' if mode == 'per_capita'
                    else f'biomass with ±flux × {dt:g} d'))
    ax.set_title(title)
    # 3-column legend keeps the 11-segment list compact.
    ax.legend(loc='upper right', fontsize=7, ncol=3, framealpha=0.95)
    ax.grid(axis='y', linestyle=':', alpha=0.4, zorder=1)

    return ax


# =================================================================
# INTRINSIC FLUX VIEW — reference biomass, not steady state
# =================================================================
#
# These functions compute and plot per-class fluxes assuming a *reference*
# biomass distribution (uniform or Sheldon-flat per linear µm) rather than
# the steady-state biomass. Bars are centred at zero (since the reference
# distribution is not a steady state); positive segments above are sources,
# negative segments below are sinks. The net per-capita rate (visible as
# the imbalance) is the model's intrinsic per-class growth rate at the
# reference biomass — i.e., which classes "would grow" if the system
# started from a flat IC.
#
# Use these to expose structural / parametric trade-offs imposed by the
# allometries, kernel, and parameters, without the masking effect of the
# equilibrium that emerges. Companion diagnostic to the SS flux plots.


def _build_reference_biomass(p_esd, z_esd, biomass_dist, biomass_ref,
                              P_ss, Z_ss, alive_P, alive_Z):
    """Construct reference P and Z biomass arrays per the chosen convention.

    biomass_dist:
      - 'equal'  : B_i = B_ref for all i. For log-spaced classes this IS the
                   Sheldon-flat reference (equal biomass per log-decade), the
                   canonical neutral baseline from Sheldon (1967, 1972) onward.
      - 'per_um' : B_i ∝ ESD_i, normalised so the geometric-mean class equals
                   B_ref. This is NOT Sheldon-flat — it has slope +1 per
                   log-decade (each size doubling carries twice the biomass).
                   Appears "flat" only when plotted with Taniguchi's per-µm
                   normalisation. Useful as a "large-class-heavy" reference
                   for sensitivity exploration; not an ecological baseline.
    biomass_ref:
      - 'ss_mean': B_ref = mean of surviving SS biomass (per group).
      - 'unit'   : B_ref = 1.0 mmol N m^-3.
    """
    n_P, n_Z = p_esd.size, z_esd.size

    if biomass_ref == 'ss_mean':
        P_ref = float(np.mean(P_ss[alive_P])) if alive_P.any() else 0.05
        Z_ref = float(np.mean(Z_ss[alive_Z])) if alive_Z.any() else 0.05
    elif biomass_ref == 'unit':
        P_ref = 1.0
        Z_ref = 1.0
    elif isinstance(biomass_ref, (int, float)):
        # Numeric: use the same value for both P and Z reference magnitudes.
        P_ref = float(biomass_ref)
        Z_ref = float(biomass_ref)
    elif (isinstance(biomass_ref, (tuple, list)) and len(biomass_ref) == 2):
        # (P_ref, Z_ref) tuple for asymmetric magnitudes.
        P_ref = float(biomass_ref[0])
        Z_ref = float(biomass_ref[1])
    else:
        raise ValueError(
            f"biomass_ref must be 'ss_mean', 'unit', a number, or a "
            f"(P_ref, Z_ref) tuple; got {biomass_ref!r}"
        )

    if biomass_dist == 'equal':
        P = np.full(n_P, P_ref)
        Z = np.full(n_Z, Z_ref)
    elif biomass_dist == 'per_um':
        # Normalise to the geometric-mean ESD so the geomean class lands at *_ref.
        p_geomean = float(np.exp(np.mean(np.log(p_esd))))
        z_geomean = float(np.exp(np.mean(np.log(z_esd))))
        P = P_ref * p_esd / p_geomean
        Z = Z_ref * z_esd / z_geomean
    else:
        raise ValueError(f"biomass_dist must be 'equal' or 'per_um', got {biomass_dist!r}")

    return P, Z, P_ref, Z_ref


# ---------------------------------------------------------------
# Model-component flux invocation helpers
# ---------------------------------------------------------------
class _MockComponentSelf:
    """Minimal stand-in for an XSO component instance, exposing only the
    `.m` math namespace (numpy) that the flux methods reference internally
    (e.g. self.m.sum, self.m.concatenate). Sufficient for calling
    @xso.flux-decorated methods outside the model integration loop.
    """
    m = np


_NS = _MockComponentSelf()


def _get_model_components():
    """Lazy import of the model's flux components.

    Imports are inside the function so flux_plot.py itself remains importable
    without `cariaco_ssm_comps` on sys.path; only the intrinsic-flux helpers
    require it. The notebook-side path setup
    (`sys.path.insert(0, os.path.abspath('../model'))`) must run before the
    intrinsic functions are called.
    """
    try:
        from cariaco_ssm_comps import (
            MonodGrowth_SizeBased,
            SizebasedGrazingMatrix_Full_TypeIII,
            PhytoMortality_toD_toN,
            ZooQuadraticMortality_toD,
            FishGrazing_Kernel,
        )
    except ImportError as e:
        raise ImportError(
            "plot_intrinsic_*_fluxes uses the model's actual flux functions "
            "as the single source of truth (no re-implementation in flux_plot.py). "
            "This requires cariaco_ssm_comps to be importable. Make sure the "
            "model folder is on sys.path before calling, e.g.:\n"
            "    import sys, os\n"
            "    sys.path.insert(0, os.path.abspath('../model'))\n"
            f"Underlying ImportError: {e}"
        )
    return {
        'monod':   MonodGrowth_SizeBased,
        'grazing': SizebasedGrazingMatrix_Full_TypeIII,
        'p_mort':  PhytoMortality_toD_toN,
        'z_mort':  ZooQuadraticMortality_toD,
        'fish':    FishGrazing_Kernel,
    }


def _call_flux(component, flux_name, **kwargs):
    """Invoke a model component's @xso.flux-decorated method directly.

    XSO wraps flux methods as keyword-only `static_flux` callables that
    inject the math backend internally — so we pass kwargs only, no `self`.
    Unused declared kwargs (e.g. `detritus` on PhytoMortality.mortality)
    accept any harmless dummy value.
    """
    flux_fn = getattr(component.fluxes, flux_name)
    return flux_fn(**kwargs)


def _compute_intrinsic_fluxes(out, biomass_dist, biomass_ref, N_ref,
                                extinct_threshold):
    """Compute all per-class P and Z fluxes at the reference biomass.

    Re-implements the model's flux formulas in pure numpy using parameters
    pulled from `out`. Returns absolute fluxes (mmol N m^-3 d^-1) and the
    reference biomass arrays so callers can derive per-capita rates.
    """
    p_esd = out['Phytoplankton__phyto_esd_index'].values
    z_esd = out['Zooplankton__zoo_esd_index'].values
    n_P, n_Z = p_esd.size, z_esd.size

    # SS values (only used for the 'ss_mean' reference)
    P_ss = np.abs(out['Phytoplankton__biomass'].isel(time=-1).values)
    Z_ss = np.abs(out['Zooplankton__biomass'].isel(time=-1).values)
    alive_P = np.isfinite(P_ss) & (P_ss > extinct_threshold)
    alive_Z = np.isfinite(Z_ss) & (Z_ss > extinct_threshold)

    P, Z, P_ref, Z_ref = _build_reference_biomass(
        p_esd, z_esd, biomass_dist, biomass_ref, P_ss, Z_ss, alive_P, alive_Z
    )

    # N reference: SS N* unless overridden.
    if N_ref is None:
        N_ref = float(out['Nutrient__value'].isel(time=-1).values)

    # ---- model parameters ----
    mu_max      = out['Growth__mu_max'].values
    K_s         = out['Growth__halfsat'].values
    rate_P_mort = out['PhytoMortality__rate'].values
    Imax        = out['Grazing__Imax'].values
    K_sZ        = float(out['Grazing__KsZ'].values)
    phiPZ       = out['Grazing__phiPZ'].values
    gge         = float(out['GGE__gge'].values)
    m_Z         = float(out['ZooMortality__rate'].values)
    F           = float(out['FishForcing__value'].values)
    r_F         = float(out['FishGrazing__rate'].values)
    K_P_fish    = out['FishGrazing__kernel_P'].values
    K_Z_fish    = out['FishGrazing__kernel_Z'].values

    # ---- compute fluxes via the actual model component flux functions ----
    # Single source of truth: any change to the model formulas in
    # cariaco_ssm_comps automatically propagates here. We bypass the
    # xsimlab solver and call the @xso.flux methods directly with a
    # minimal stand-in for `self` that exposes only `.m = numpy` (the
    # math namespace the flux bodies use, e.g. self.m.sum, self.m.concatenate).
    comps = _get_model_components()

    U      = _call_flux(comps['monod'], 'uptake',
                        resource=N_ref, consumer=P,
                        halfsat=K_s, mu_max=mu_max)
    G_full = _call_flux(comps['grazing'], 'grazing',
                        resource=P, consumer=Z,
                        phiPZ=phiPZ, Imax=Imax, KsZ=K_sZ)
    # PhytoMortality.mortality body uses only `population` and `rate`;
    # the other declared kwargs are passed as harmless dummies.
    M_P    = _call_flux(comps['p_mort'], 'mortality',
                        population=P, detritus=0.0, nutrient=0.0,
                        rate=rate_P_mort, f_mort_D=0.5)
    # ZooQuadraticMortality.mortality body uses `population`, `rate`, and
    # self.m.sum(population); other kwargs are dummies.
    M_Z    = _call_flux(comps['z_mort'], 'mortality',
                        population=Z, detritus=0.0,
                        rate=m_Z, f_mort_D=0.5)
    Ff_P   = _call_flux(comps['fish'], 'fish_graze_phyto',
                        phyto=P, zoo=Z, fish_forcing=F,
                        kernel_P=K_P_fish, kernel_Z=K_Z_fish, rate=r_F)
    Ff_Z   = _call_flux(comps['fish'], 'fish_graze_zoo',
                        phyto=P, zoo=Z, fish_forcing=F,
                        kernel_P=K_P_fish, kernel_Z=K_Z_fish, rate=r_F)

    return {
        'P_ref':    P,
        'Z_ref':    Z,
        'P_geomean_ref': P_ref,
        'Z_geomean_ref': Z_ref,
        'N_ref':    N_ref,
        # P
        'P_uptake':              U,
        'P_graze_per_predator':  G_full[:n_P, :],
        'P_mortality':           M_P,
        'P_fish':                Ff_P,
        # Z
        'Z_assim_per_prey':      gge * G_full,
        'Z_graze_per_predator':  G_full[n_P:, :],
        'Z_closure':             M_Z,
        'Z_fish':                Ff_Z,
        # Auxiliaries
        'phiPZ':         phiPZ,
        'p_esd':         p_esd,
        'z_esd':         z_esd,
        'gge':           gge,
        'biomass_dist':  biomass_dist,
        'biomass_ref':   biomass_ref,
    }


def _intrinsic_label(f):
    """Compact human-readable description of the reference state."""
    dist = {'equal': 'equal/log-decade',
            'per_um': 'per linear µm'}[f['biomass_dist']]
    bref = f['biomass_ref']
    if bref == 'ss_mean':
        ref = 'SS-mean'
    elif bref == 'unit':
        ref = '1.0'
    elif isinstance(bref, (int, float)):
        ref = f'{float(bref):.3g}'
    elif isinstance(bref, (tuple, list)) and len(bref) == 2:
        ref = f'P={float(bref[0]):.3g}/Z={float(bref[1]):.3g}'
    else:
        ref = str(bref)
    return (f"biomass dist: {dist}, B_ref({ref}) "
            f"P={f['P_geomean_ref']:.2g} / Z={f['Z_geomean_ref']:.2g}, "
            f"N={f['N_ref']:.3g}")


# ---------------------------------------------------------------
# Intrinsic P plot
# ---------------------------------------------------------------
def plot_intrinsic_phyto_fluxes(
    out,
    *,
    biomass_dist: str = 'equal',
    biomass_ref: str = 'ss_mean',
    N_ref: float | None = None,
    extinct_threshold: float = 1e-7,
    bar_width: float = 0.7,
    show_size_bands: bool = True,
    ax=None,
    title: str | None = None,
):
    """Per-class P per-capita flux structure at a reference biomass distribution.

    Bars are centred at zero (the reference state is NOT a steady state).
    Positive segments above = per-capita uptake; negative segments below =
    per-capita losses (3-group grazing decomposition + mortality + fish).
    The visible imbalance is the model's intrinsic per-class net per-capita
    growth rate at the reference state — positive = "this class would grow",
    negative = "this class would shrink".

    Parameters
    ----------
    out : xarray.Dataset
        XSO model output (used for parameters and, if `biomass_ref='ss_mean'`,
        for the SS biomass mean reference).
    biomass_dist : {'equal', 'per_um'}
        - 'equal'  : B_i = B_ref for all i (Sheldon-flat per log-decade).
        - 'per_um' : B_i ∝ ESD_i (Sheldon-flat per linear µm, Taniguchi).
    biomass_ref : {'ss_mean', 'unit'}
        - 'ss_mean': B_ref = mean of surviving SS biomass.
        - 'unit'   : B_ref = 1.0 mmol N m^-3.
    N_ref : float or None
        Reference nutrient concentration. Defaults to SS N* from `out`.
    """
    f = _compute_intrinsic_fluxes(out, biomass_dist, biomass_ref, N_ref,
                                    extinct_threshold)
    P = f['P_ref']
    p_esd = f['p_esd']
    n_P = P.size
    invP = 1.0 / P

    # Per-capita rates
    pc_uptake = f['P_uptake'] * invP
    pc_mort   = f['P_mortality'] * invP
    pc_fish   = f['P_fish']      * invP

    G = f['P_graze_per_predator']                                  # (n_P, n_Z)
    groups, _ = _classify_predator_groups(f['phiPZ'][:n_P, :])
    pc_g_small = np.where(groups == 0, G, 0.0).sum(axis=1) * invP
    pc_g_opt   = np.where(groups == 1, G, 0.0).sum(axis=1) * invP
    pc_g_large = np.where(groups == 2, G, 0.0).sum(axis=1) * invP

    centre = np.zeros(n_P)
    positive_terms = [('Uptake', pc_uptake, COLOR_UPTAKE)]
    negative_terms = [
        ('Graze (Z<opt)', pc_g_small, COLOR_GRAZE_SMALL),
        ('Graze (Z=opt)', pc_g_opt,   COLOR_GRAZE_OPT),
        ('Graze (Z>opt)', pc_g_large, COLOR_GRAZE_LARGE),
        ('Linear mort.',  pc_mort,    COLOR_MORT),
        ('Fish grazing',  pc_fish,    COLOR_FISH),
    ]

    if ax is None:
        fig, ax = plt.subplots(figsize=(11.5, 6.5))
    else:
        fig = ax.figure
    x = np.arange(n_P)

    # Pico/Nano/Micro size bands
    if show_size_bands:
        def esd_to_x(esd):
            return np.interp(np.log10(esd), np.log10(p_esd), x)
        x_pn, x_nm = esd_to_x(2.0), esd_to_x(20.0)
        x_lo, x_hi = -0.5, n_P - 0.5
        ax.axvspan(x_lo, x_pn, color=COLOR_PICO,  alpha=0.7, zorder=0)
        ax.axvspan(x_pn, x_nm, color=COLOR_NANO,  alpha=0.7, zorder=0)
        ax.axvspan(x_nm, x_hi, color=COLOR_MICRO, alpha=0.7, zorder=0)
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        for x_mid, lab in [((x_lo + x_pn) / 2, 'Pico'),
                           ((x_pn + x_nm) / 2, 'Nano'),
                           ((x_nm + x_hi) / 2, 'Micro')]:
            ax.text(x_mid, 0.97, lab, transform=trans,
                    ha='center', va='top', fontsize=9, color='dimgrey', alpha=0.8)

    # Bars — all classes drawn (no extinct concept here)
    bottom = centre.copy()
    for label, values, color in positive_terms:
        ax.bar(x, values, bottom=bottom, width=bar_width,
               color=color, edgecolor='black', linewidth=0.3,
               label=label, zorder=3)
        bottom = bottom + values
    top = centre.copy()
    for label, values, color in negative_terms:
        ax.bar(x, -values, bottom=top, width=bar_width,
               color=color, edgecolor='black', linewidth=0.3,
               label=label, zorder=3)
        top = top - values

    # Net per-capita rate marker (the imbalance)
    net = pc_uptake - (pc_g_small + pc_g_opt + pc_g_large + pc_mort + pc_fish)
    ax.plot(x, net, 'k_', markersize=14, markeredgewidth=2.0,
            zorder=6, label='Net per-capita rate')

    # Zero line
    ax.axhline(0.0, color='black', linewidth=0.8, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels([f'{e:.2g}' for e in p_esd], rotation=45, ha='right')
    ax.set_xlabel('Phyto ESD (µm)')
    ax.set_ylabel('Per-capita rate (d$^{-1}$)')
    ax.set_xlim(-0.5, n_P - 0.5)

    if title is None:
        title = (f'Intrinsic per-class P fluxes — {_intrinsic_label(f)}')
    ax.set_title(title, fontsize=10)
    ax.legend(loc='upper right', fontsize=8, ncol=2, framealpha=0.95)
    ax.grid(axis='y', linestyle=':', alpha=0.4, zorder=1)

    return ax


# ---------------------------------------------------------------
# Intrinsic Z plot
# ---------------------------------------------------------------
def plot_intrinsic_zoo_fluxes(
    out,
    *,
    biomass_dist: str = 'equal',
    biomass_ref: str = 'ss_mean',
    N_ref: float | None = None,
    extinct_threshold: float = 1e-7,
    bar_width: float = 0.7,
    show_size_bands: bool = True,
    ax=None,
    title: str | None = None,
):
    """Per-class Z per-capita flux structure at a reference biomass distribution.

    Source side: 6 segments (P / Z prey × small / opt / large) shown above 0.
    Sink side: 3 predator groups + closure + fish, shown below 0.
    The net imbalance is the model's intrinsic per-class net per-capita
    growth rate at the reference state.
    """
    f = _compute_intrinsic_fluxes(out, biomass_dist, biomass_ref, N_ref,
                                    extinct_threshold)
    P = f['P_ref']
    Z = f['Z_ref']
    p_esd = f['p_esd']
    z_esd = f['z_esd']
    n_P, n_Z = P.size, Z.size
    invZ = 1.0 / Z

    # ---- Source side (6 segments) ----
    prey_esd = np.concatenate([p_esd, z_esd])
    is_phyto = np.arange(n_P + n_Z) < n_P
    prey_groups, _ = _classify_prey_groups(f['phiPZ'], prey_esd)
    A_per_pp = f['Z_assim_per_prey']                                # (n_P+n_Z, n_Z)

    def src_pc(filter_, gid):
        mask = (prey_groups == gid) & filter_[:, None]
        return (A_per_pp * mask).sum(axis=0) * invZ

    pc_src_P_small = src_pc(is_phyto,  0)
    pc_src_P_opt   = src_pc(is_phyto,  1)
    pc_src_P_large = src_pc(is_phyto,  2)
    pc_src_Z_small = src_pc(~is_phyto, 0)
    pc_src_Z_opt   = src_pc(~is_phyto, 1)
    pc_src_Z_large = src_pc(~is_phyto, 2)

    # ---- Sink side ----
    G_on_Z = f['Z_graze_per_predator']                              # (n_Z, n_Z)
    pred_groups, _ = _classify_predator_groups(f['phiPZ'][n_P:, :])
    pc_g_small = np.where(pred_groups == 0, G_on_Z, 0.0).sum(axis=1) * invZ
    pc_g_opt   = np.where(pred_groups == 1, G_on_Z, 0.0).sum(axis=1) * invZ
    pc_g_large = np.where(pred_groups == 2, G_on_Z, 0.0).sum(axis=1) * invZ
    pc_clos    = f['Z_closure'] * invZ
    pc_fish    = f['Z_fish']    * invZ

    centre = np.zeros(n_Z)
    positive_terms = [
        ('Assim P<opt', pc_src_P_small, COLOR_SRC_P_SMALL),
        ('Assim P=opt', pc_src_P_opt,   COLOR_SRC_P_OPT),
        ('Assim P>opt', pc_src_P_large, COLOR_SRC_P_LARGE),
        ('Assim Z<opt', pc_src_Z_small, COLOR_SRC_Z_SMALL),
        ('Assim Z=opt', pc_src_Z_opt,   COLOR_SRC_Z_OPT),
        ('Assim Z>opt', pc_src_Z_large, COLOR_SRC_Z_LARGE),
    ]
    negative_terms = [
        ('Pred Z<opt', pc_g_small, COLOR_GRAZE_SMALL),
        ('Pred Z=opt', pc_g_opt,   COLOR_GRAZE_OPT),
        ('Pred Z>opt', pc_g_large, COLOR_GRAZE_LARGE),
        ('Closure',    pc_clos,    COLOR_MORT),
        ('Fish',       pc_fish,    COLOR_FISH),
    ]

    if ax is None:
        fig, ax = plt.subplots(figsize=(12.5, 7.0))
    else:
        fig = ax.figure
    x = np.arange(n_Z)

    if show_size_bands:
        def esd_to_x(esd):
            return np.interp(np.log10(esd), np.log10(z_esd), x)
        x_200, x_500 = esd_to_x(200.0), esd_to_x(500.0)
        x_lo, x_hi = -0.5, n_Z - 0.5
        ax.axvspan(x_lo,  x_200, color=COLOR_BAND_MICROZOO, alpha=0.7, zorder=0)
        ax.axvspan(x_200, x_500, color=COLOR_BAND_MESO_LO,  alpha=0.7, zorder=0)
        ax.axvspan(x_500, x_hi,  color=COLOR_BAND_MESO_HI,  alpha=0.7, zorder=0)
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        for x_mid, lab in [((x_lo + x_200) / 2, 'Microzoo (sub-obs)'),
                           ((x_200 + x_500) / 2, 'Zoo (200, 500]'),
                           ((x_500 + x_hi)  / 2, 'Zoo > 500')]:
            ax.text(x_mid, 0.97, lab, transform=trans,
                    ha='center', va='top', fontsize=9, color='dimgrey', alpha=0.8)

    # All classes drawn
    bottom = centre.copy()
    for label, values, color in positive_terms:
        ax.bar(x, values, bottom=bottom, width=bar_width,
               color=color, edgecolor='black', linewidth=0.3,
               label=label, zorder=3)
        bottom = bottom + values
    top = centre.copy()
    for label, values, color in negative_terms:
        ax.bar(x, -values, bottom=top, width=bar_width,
               color=color, edgecolor='black', linewidth=0.3,
               label=label, zorder=3)
        top = top - values

    total_src_pc  = (pc_src_P_small + pc_src_P_opt + pc_src_P_large
                     + pc_src_Z_small + pc_src_Z_opt + pc_src_Z_large)
    total_loss_pc = pc_g_small + pc_g_opt + pc_g_large + pc_clos + pc_fish
    net_pc = total_src_pc - total_loss_pc
    ax.plot(x, net_pc, 'k_', markersize=14, markeredgewidth=2.0,
            zorder=6, label='Net per-capita rate')
    ax.axhline(0.0, color='black', linewidth=0.8, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels([f'{e:.4g}' for e in z_esd], rotation=45, ha='right')
    ax.set_xlabel('Zoo ESD (µm)')
    ax.set_ylabel('Per-capita rate (d$^{-1}$)')
    ax.set_xlim(-0.5, n_Z - 0.5)

    if title is None:
        title = (f'Intrinsic per-class Z fluxes — {_intrinsic_label(f)}')
    ax.set_title(title, fontsize=10)
    ax.legend(loc='upper right', fontsize=7, ncol=3, framealpha=0.95)
    ax.grid(axis='y', linestyle=':', alpha=0.4, zorder=1)

    return ax


# ---------------------------------------------------------------
# Usage example
# ---------------------------------------------------------------
if __name__ == '__main__':
    # Example:
    #   import xarray as xr
    #   import xso  # required for the xsimlab accessor
    #   out = xr.open_dataset('best_cell_steady_state.nc')
    #
    #   # Phyto:
    #   ax = plot_phyto_flux_decomposition(out, mode='biomass', dt=1.0)
    #   plt.tight_layout(); plt.show()
    #   summarize_phyto_fluxes(out)
    #
    #   # Zoo:
    #   ax = plot_zoo_flux_decomposition(out, mode='biomass', dt=1.0)
    #   plt.tight_layout(); plt.show()
    #   summarize_zoo_fluxes(out)
    pass
