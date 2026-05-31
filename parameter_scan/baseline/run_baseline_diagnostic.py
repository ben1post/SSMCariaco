"""
Baseline diagnostic runner — Option A
=====================================
Run-and-plot script for the iteration-1 baseline NPZ model defined in
`cariaco_baseline_comps.py` + `cariaco_baseline_setups.py`.

Exposes three functions for notebook use:
- `run_setup(model, setup)`    : execute an IVP setup, return xarray Dataset
- `extract_state(out)`         : pull N, P, Z arrays in (dim, time) layout
- `plot_diagnostic(...)`       : 4-panel diagnostic figure (N timeseries,
                                  P timeseries, Z timeseries, size spectrum)

Run as a script for default Option A baseline diagnostic at FN_DEFAULT:
    python run_baseline_diagnostic.py
"""

import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# RUNNER + EXTRACTION
# =============================================================================
def run_setup(model, setup, input_vars_override=None):
    """Execute an XSO setup under its model context, optionally with overrides.

    Parameters
    ----------
    model : xso model object (from xso.create)
    setup : xarray.Dataset (from xso.setup)
    input_vars_override : dict or None, optional
        Per-component input_vars overrides to apply at run time, e.g.
        ``{'Supply': {'FN': 8.0}}``. xsimlab's ``update_vars`` requires the
        model to be in the active context, so applying it inside ``with model:``
        is the simplest way to chain a parameter override onto a base setup
        without rebuilding the full setup object.

    Returns
    -------
    out : xarray.Dataset with state-variable and flux outputs.
    """
    with model:
        if input_vars_override is not None:
            setup = setup.xsimlab.update_vars(input_vars=input_vars_override)
        out = setup.xsimlab.run()
    return out


def extract_state(out, n_label='Nutrient__value',
                  p_label='Phytoplankton__biomass',
                  z_label='Zooplankton__biomass'):
    """Pull N, P, Z arrays from an XSO output dataset.

    Per XSO_HANDOFF.md:1098, output is (dim, time) for dimensioned variables.

    Returns
    -------
    state : dict with keys 'N', 'P', 'Z', 't'.
            N has shape (n_time,); P has (n_phyto, n_time); Z has (n_zoo, n_time).
    """
    N = out[n_label].values                          # (n_time,)
    P = out[p_label].values                          # (n_phyto, n_time)
    Z = out[z_label].values                          # (n_zoo,   n_time)
    t = out['time'].values if 'time' in out.coords else np.arange(len(N))
    return dict(N=N, P=P, Z=Z, t=t)


def summarise_tail(state, tail_len=1000):
    """Compute tail-mean per-class biomass + simple summary stats.

    Parameters
    ----------
    state : dict from extract_state()
    tail_len : int, number of trailing time steps to average over.
    """
    P, Z, N = state['P'], state['Z'], state['N']
    P_tail = P[:, -tail_len:].mean(axis=1)
    Z_tail = Z[:, -tail_len:].mean(axis=1)
    N_tail = N[-tail_len:].mean()
    return dict(
        P_tail=P_tail,
        Z_tail=Z_tail,
        N_tail=N_tail,
        sumP=P_tail.sum(),
        sumZ=Z_tail.sum(),
        ZP_ratio=Z_tail.sum() / max(P_tail.sum(), 1e-12),
        n_P_alive=int((P_tail > 1e-6).sum()),
        n_Z_alive=int((Z_tail > 1e-6).sum()),
    )


# =============================================================================
# 4-PANEL DIAGNOSTIC PLOT
# =============================================================================
def plot_diagnostic(state, phyto_esd, zoo_esd, FN, de,
                    title_suffix='', tail_len=1000,
                    save_path=None, figsize=(13, 9),
                    sieburth_bands=True):
    """Render a 4-panel diagnostic of an Option A run.

    Panels:
        (a) N(t)
        (b) per-class P(t), log y, viridis colourmap
        (c) per-class Z(t), log y, plasma colourmap
        (d) tail-mean size spectrum, P+Z overlaid, log-log,
            Sieburth Pico/Nano/Micro bands marked

    Parameters
    ----------
    state : dict from extract_state()
    phyto_esd, zoo_esd : 1D arrays of size-class ESD (µm)
    FN, de : forcing values for the run, used in panel titles only
    title_suffix : str, appended to the suptitle
    tail_len : int, trailing window for the size-spectrum mean
    save_path : str or None, saves to file if given
    sieburth_bands : bool, shade Pico/Nano/Micro extents in panel (d)
    """
    P, Z, N, t = state['P'], state['Z'], state['N'], state['t']
    n_classes_P = P.shape[0]
    n_classes_Z = Z.shape[0]

    fig, ax = plt.subplots(2, 2, figsize=figsize)

    # (a) Nutrient
    ax[0, 0].plot(t, N, color='k', lw=1.2)
    ax[0, 0].set_xlabel('Time [d]')
    ax[0, 0].set_ylabel('N [mmol N m⁻³]')
    ax[0, 0].set_title(f'(a) Nutrient  (F_N = {FN}, d_e = {de})')
    ax[0, 0].grid(alpha=0.3)

    # (b) Phyto per class
    cP = plt.cm.viridis(np.linspace(0, 1, n_classes_P))
    label_idx_P = sorted(set([0, n_classes_P // 2, n_classes_P - 1]))
    for i in range(n_classes_P):
        lab = f'{phyto_esd[i]:.2f} µm' if i in label_idx_P else None
        ax[0, 1].plot(t, P[i], color=cP[i], lw=0.8, label=lab)
    ax[0, 1].set_xlabel('Time [d]')
    ax[0, 1].set_ylabel('P [mmol N m⁻³]')
    ax[0, 1].set_yscale('log')
    ax[0, 1].set_title('(b) Phytoplankton biomass per class')
    ax[0, 1].legend(fontsize=8, loc='best')
    ax[0, 1].grid(alpha=0.3, which='both')

    # (c) Zoo per class
    cZ = plt.cm.plasma(np.linspace(0, 1, n_classes_Z))
    label_idx_Z = sorted(set([0, n_classes_Z // 2, n_classes_Z - 1]))
    for i in range(n_classes_Z):
        lab = f'{zoo_esd[i]:.1f} µm' if i in label_idx_Z else None
        ax[1, 0].plot(t, Z[i], color=cZ[i], lw=0.8, label=lab)
    ax[1, 0].set_xlabel('Time [d]')
    ax[1, 0].set_ylabel('Z [mmol N m⁻³]')
    ax[1, 0].set_yscale('log')
    ax[1, 0].set_title('(c) Zooplankton biomass per class')
    ax[1, 0].legend(fontsize=8, loc='best')
    ax[1, 0].grid(alpha=0.3, which='both')

    # (d) Tail-mean size spectrum
    P_tail = P[:, -tail_len:].mean(axis=1)
    Z_tail = Z[:, -tail_len:].mean(axis=1)
    if sieburth_bands:
        ax[1, 1].axvspan(0.2,    2.0, alpha=0.08, color='steelblue', label='Pico')
        ax[1, 1].axvspan(2.0,   20.0, alpha=0.08, color='gold',      label='Nano')
        ax[1, 1].axvspan(20.0, 200.0, alpha=0.08, color='salmon',    label='Micro')
    ax[1, 1].plot(phyto_esd, P_tail, 'o-', color='seagreen',
                  lw=1.5, ms=7, label='P tail-mean')
    ax[1, 1].plot(zoo_esd,   Z_tail, 's-', color='firebrick',
                  lw=1.5, ms=7, label='Z tail-mean')
    ax[1, 1].set_xlabel('ESD [µm]')
    ax[1, 1].set_ylabel('Biomass [mmol N m⁻³]')
    ax[1, 1].set_xscale('log')
    ax[1, 1].set_yscale('log')
    ax[1, 1].set_title(f'(d) Size spectrum (last {tail_len} d mean)')
    ax[1, 1].legend(fontsize=9, loc='best')
    ax[1, 1].grid(alpha=0.3, which='both')

    title = 'Option A baseline — Taniguchi M1 biology + Stock supply + sinking (no D, no fish)'
    if title_suffix:
        title += f'\n{title_suffix}'
    plt.suptitle(title, fontsize=12, y=1.005)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=140, bbox_inches='tight')
        print(f'Saved figure to: {save_path}')

    return fig


# =============================================================================
# SLOPE METRICS — bridge to parscan_utils / cariaco_obs (Sieburth bins)
# =============================================================================
def compute_metrics(state, phyto_esd, zoo_esd, tail_len=1000):
    """Compute Cariaco-bin metrics on the tail-mean spectrum.

    Uses parscan_utils.aggregate_model_to_targets + compute_phyto_spectrum_metrics
    on the standard 3 Sieburth phyto bins (Pico/Nano/Micro from
    cariaco_obs.TARGET_BIN_DEFINITIONS).

    Returns
    -------
    metrics : dict with 'centroid', 'shannon', 'slope_2pt', 'nbss_slope',
              'P_bins' (Pico/Nano/Micro biomass triple), 'fractions'.
    """
    from cariaco_obs import TARGET_BIN_DEFINITIONS
    from parscan_utils import (aggregate_model_to_targets,
                               compute_phyto_spectrum_metrics)

    phyto_bin_defs = [b for b in TARGET_BIN_DEFINITIONS if b['type'] == 'phyto']
    P_tail = state['P'][:, -tail_len:].mean(axis=1)
    model_vec = aggregate_model_to_targets(
        {'phyto': P_tail}, phyto_esd, zoo_esd, phyto_bin_defs)
    m = compute_phyto_spectrum_metrics(model_vec, phyto_bin_defs)
    m['P_bins'] = model_vec    # [Pico, Nano, Micro] biomass
    return m


# =============================================================================
# OBS VALIDATION — single-point model vs Cariaco obs targets
# =============================================================================
def obs_comparison_table(out, phyto_esd, zoo_esd, regime='all', tail_len=1000,
                         pp_var='Growth__uptake_value',
                         export_var='PhytoSinking__sinking_value',
                         de_var='Supply__de'):
    """Compare tail-mean model output to Cariaco obs targets.

    Aggregates an Option A IVP output dataset across the 8 target bins from
    cariaco_obs (3 phyto, 2 zoo, NO3, PP, Export) via
    parscan_utils.aggregate_model_to_targets, and prints a side-by-side
    model-vs-obs table with model/obs ratio.

    Pulls PP and export from per-class flux outputs in the dataset:
        PP        : tail-mean of `pp_var`     (default 'Growth__uptake_value')
        Export    : tail-mean of `export_var` (default 'PhytoSinking__sinking_value')
        d_e       : scalar from `de_var`      (default 'Supply__de')

    Per parscan_utils.aggregate_model_to_targets logic:
        Export is converted from volumetric (mmol N m-3 d-1) to areal
        (mmol N m-2 d-1) by multiplying with d_e — matches Cariaco trap obs.

    Parameters
    ----------
    out : xarray.Dataset — full XSO output (not just the state dict), needed
          for flux variables and the d_e parameter.
    phyto_esd, zoo_esd : 1D arrays of size-class ESD.
    regime : 'all' | 'upwelling' | 'relaxed' | 'strong' | 'moderate' | 'weak'
             — passed to cariaco_obs.load_cariaco_targets.
    tail_len : int, trailing window for the tail-mean averaging.

    Returns
    -------
    df : pandas DataFrame with columns
         ['label', 'model', 'obs', 'ratio', 'regime_FN', 'regime_de'].
    """
    import pandas as pd
    from cariaco_obs import load_cariaco_targets
    from parscan_utils import aggregate_model_to_targets

    obs_vec, labels, bin_defs, _, forcing = load_cariaco_targets(regime=regime)

    # State (tail-means)
    P_tail = out['Phytoplankton__biomass'].values[:, -tail_len:].mean(axis=1)
    Z_tail = out['Zooplankton__biomass'].values[:, -tail_len:].mean(axis=1)
    N_tail = float(out['Nutrient__value'].values[-tail_len:].mean())

    # PP per class (mmol N m-3 d-1); aggregate function sums internally
    PP_tail = out[pp_var].values[:, -tail_len:].mean(axis=1)

    # Phyto sinking (per-class volumetric flux) summed to scalar; the aggregate
    # function multiplies by d_e to convert to areal mmol N m-2 d-1
    sink_tail = out[export_var].values[:, -tail_len:].mean(axis=1)
    d_e = float(out[de_var].values)

    model_state = {
        'phyto':    P_tail,
        'zoo':      Z_tail,
        'nutrient': N_tail,
        'pp':       PP_tail,                # per-class uptake [mmol N m-3 d-1]
        'export':   float(sink_tail.sum()), # total volumetric [mmol N m-3 d-1]
        'd_e':      d_e,                    # for areal-export conversion
    }
    model_vec = aggregate_model_to_targets(
        model_state, phyto_esd, zoo_esd, bin_defs)

    df = pd.DataFrame({
        'label': labels,
        'model': model_vec,
        'obs':   obs_vec,
    })
    df['ratio'] = df['model'] / df['obs'].replace(0, np.nan)
    df['regime_FN'] = forcing['Inflow__FN']
    df['regime_de'] = forcing['Inflow__de']
    return df


# =============================================================================
# STABILITY DIAGNOSTICS — per-class CV + regime classification
# =============================================================================
# Stability is measured by the coefficient of variation (CV = std/|mean|) over
# the tail window per class, restricted to "active" classes (tail-mean above
# a floor). The max CV across active classes determines the dynamical regime
# label, following the May 22 2026 bridge-figure convention:
#   CV < mild_threshold (0.05)   → fixed point (settled)
#   CV < big_threshold  (0.50)   → mild limit cycle
#   CV ≥ big_threshold           → large-amplitude limit cycle / chaos

def class_cv(arr, tail_len=1000, floor=1e-6):
    """Per-class coefficient of variation over the tail.

    Parameters
    ----------
    arr : ndarray of shape (n_classes, n_time).
    tail_len : int, trailing window for the tail.
    floor : float, classes with tail-mean below this are returned as NaN
            (oscillation in numerically-extinct classes is solver noise,
            not biological signal).

    Returns
    -------
    mean : (n_classes,) tail-mean per class
    cv   : (n_classes,) tail-CV per class; NaN where mean < floor
    """
    tail = arr[:, -tail_len:]
    mean = tail.mean(axis=1)
    std  = tail.std(axis=1)
    cv   = np.where(mean > floor, std / np.abs(mean), np.nan)
    return mean, cv


def regime_from_cv(cv, mild_threshold=0.05, big_threshold=0.5):
    """Classify dynamical regime by max CV across active classes.

    Returns
    -------
    label : str, regime description with the max CV inline
    max_cv : float (or NaN if no active classes)
    """
    active = cv[np.isfinite(cv)]
    if len(active) == 0:
        return 'no alive classes', float('nan')
    max_cv = float(np.nanmax(active))
    if max_cv < mild_threshold:
        label = f'fixed point (max CV = {max_cv:.3f})'
    elif max_cv < big_threshold:
        label = f'mild limit cycle (max CV = {max_cv:.3f})'
    else:
        label = f'large-amplitude / chaos (max CV = {max_cv:.3f})'
    return label, max_cv


def stability_summary(state, tail_len=1000, verbose=True):
    """Compute and (optionally) print P/Z stability metrics for a state.

    Returns
    -------
    summary : dict with keys
        'mP', 'cvP' — tail-mean and CV per phyto class
        'mZ', 'cvZ' — tail-mean and CV per zoo class
        'regime_P', 'regime_Z' — string labels
        'max_cv_P', 'max_cv_Z' — float
    """
    mP, cvP = class_cv(state['P'], tail_len=tail_len)
    mZ, cvZ = class_cv(state['Z'], tail_len=tail_len)
    regime_P, max_cv_P = regime_from_cv(cvP)
    regime_Z, max_cv_Z = regime_from_cv(cvZ)
    if verbose:
        print(f'  P: {regime_P}')
        print(f'  Z: {regime_Z}')
    return dict(
        mP=mP, cvP=cvP, mZ=mZ, cvZ=cvZ,
        regime_P=regime_P, regime_Z=regime_Z,
        max_cv_P=max_cv_P, max_cv_Z=max_cv_Z,
    )


def plot_dynamics_compare_pair(state_a, state_b, label_a, label_b,
                                phyto_esd, zoo_esd, fn_value,
                                tail_len=1000, zoom_len=500,
                                save_path=None, figsize=(13, 11)):
    """Side-by-side dynamics comparison of TWO model variants at one F_N.

    Layout: 3 rows × 2 cols.
        Row 1: N(t) full timeseries, variant A | variant B
        Row 2: ΣP, ΣZ — zoom to last `zoom_len` days, A | B
        Row 3: per-class CV bar charts — P side | Z side, A vs B grouped

    For more than 2 variants, call this function pairwise or use
    `scan_FN_baseline.scan_FN_stability` for the multi-F_N tabulation.
    """
    fig, ax = plt.subplots(3, 2, figsize=figsize)
    t = state_a['t']
    zoom = slice(-zoom_len, None)

    # Row 1 — N(t) full
    for col, (lab, st) in enumerate(zip([label_a, label_b],
                                          [state_a, state_b])):
        ax[0, col].plot(st['t'], st['N'], color='black', lw=1.0)
        ax[0, col].set_xlabel('Time [d]')
        ax[0, col].set_ylabel('N [mmol N m⁻³]')
        ax[0, col].set_title(f'(a{col+1}) N(t) — {lab}')
        ax[0, col].grid(alpha=0.3)

    # Row 2 — ΣP, ΣZ zoom
    for col, (lab, st) in enumerate(zip([label_a, label_b],
                                          [state_a, state_b])):
        sumP = st['P'].sum(axis=0)
        sumZ = st['Z'].sum(axis=0)
        ax[1, col].plot(st['t'][zoom], sumP[zoom],
                         label='ΣP', color='seagreen',  lw=1.2)
        ax[1, col].plot(st['t'][zoom], sumZ[zoom],
                         label='ΣZ', color='firebrick', lw=1.2)
        ax[1, col].set_xlabel('Time [d]')
        ax[1, col].set_ylabel('Biomass [mmol N m⁻³]')
        ax[1, col].set_title(f'(b{col+1}) ΣP, ΣZ (last {zoom_len} d) — {lab}')
        ax[1, col].legend(fontsize=8)
        ax[1, col].grid(alpha=0.3)

    # Row 3 — per-class CV bar charts (P side col 0, Z side col 1)
    _, cvP_a = class_cv(state_a['P'], tail_len)
    _, cvP_b = class_cv(state_b['P'], tail_len)
    _, cvZ_a = class_cv(state_a['Z'], tail_len)
    _, cvZ_b = class_cv(state_b['Z'], tail_len)

    n_cls = len(phyto_esd)
    idx = np.arange(n_cls)
    w = 0.35

    ax[2, 0].bar(idx - w/2, cvP_a, w, label=label_a,
                  color='steelblue',  edgecolor='black', lw=0.4)
    ax[2, 0].bar(idx + w/2, cvP_b, w, label=label_b,
                  color='goldenrod',  edgecolor='black', lw=0.4)
    ax[2, 0].axhline(0.05, color='red', ls='--', lw=0.9, label='CV=0.05')
    ax[2, 0].axhline(0.50, color='maroon', ls=':', lw=0.9, label='CV=0.50')
    ax[2, 0].set_xticks(idx)
    ax[2, 0].set_xticklabels([f'{e:.1f}' for e in phyto_esd],
                               rotation=60, fontsize=8)
    ax[2, 0].set_xlabel('Phyto ESD [µm]')
    ax[2, 0].set_ylabel('CV of P_i over tail')
    ax[2, 0].set_title('(c) Per-class P stability (CV)')
    ax[2, 0].legend(fontsize=8); ax[2, 0].grid(alpha=0.3, axis='y')

    ax[2, 1].bar(idx - w/2, cvZ_a, w, label=label_a,
                  color='steelblue',  edgecolor='black', lw=0.4)
    ax[2, 1].bar(idx + w/2, cvZ_b, w, label=label_b,
                  color='goldenrod',  edgecolor='black', lw=0.4)
    ax[2, 1].axhline(0.05, color='red', ls='--', lw=0.9, label='CV=0.05')
    ax[2, 1].axhline(0.50, color='maroon', ls=':', lw=0.9, label='CV=0.50')
    ax[2, 1].set_xticks(idx)
    ax[2, 1].set_xticklabels([f'{e:.0f}' for e in zoo_esd],
                               rotation=60, fontsize=8)
    ax[2, 1].set_xlabel('Zoo ESD [µm]')
    ax[2, 1].set_ylabel('CV of Z_i over tail')
    ax[2, 1].set_title('(d) Per-class Z stability (CV)')
    ax[2, 1].legend(fontsize=8); ax[2, 1].grid(alpha=0.3, axis='y')

    plt.suptitle(f'Dynamics comparison at F_N = {fn_value} '
                  f'(tail window = {tail_len} d)',
                  fontsize=12, y=1.005)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=140, bbox_inches='tight')
        print(f'Saved figure to: {save_path}')
    return fig


def print_summary(state, tail_len=1000, n_classes_label='12'):
    """Print a one-glance summary of an Option A run to stdout."""
    s = summarise_tail(state, tail_len=tail_len)
    print(f'\nTail-mean summary (last {tail_len} d):')
    print(f'  N             : {s["N_tail"]:.4f} mmol N m-3')
    print(f'  ΣP            : {s["sumP"]:.4f} mmol N m-3')
    print(f'  ΣZ            : {s["sumZ"]:.4f} mmol N m-3')
    print(f'  Z:P ratio     : {s["ZP_ratio"]:.3f}')
    print(f'  Alive P (>1e-6): {s["n_P_alive"]} / {n_classes_label}')
    print(f'  Alive Z (>1e-6): {s["n_Z_alive"]} / {n_classes_label}')
    return s


# =============================================================================
# MAIN — DEFAULT BASELINE DIAGNOSTIC
# =============================================================================
def main(save_path='fig_baseline_optA_FN267.png'):
    """Default end-to-end run of the Option A baseline diagnostic."""
    import cariaco_baseline_setups as cbs

    out = run_setup(cbs.model_baseline, cbs.model_setup_baseline)
    state = extract_state(out)
    fig = plot_diagnostic(
        state,
        phyto_esd=cbs.phyto_esd, zoo_esd=cbs.zoo_esd,
        FN=cbs.FN_DEFAULT, de=cbs.DE_DEFAULT,
        save_path=save_path,
    )
    print_summary(state, n_classes_label=str(cbs.N_CLASSES))
    plt.show()
    return out, state, fig


if __name__ == '__main__':
    main()
