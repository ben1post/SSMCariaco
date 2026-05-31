"""
F_N scan for the iteration-1 baseline (Option A) and fish variant.
==================================================================
Loops over F_N values, runs the Option A baseline IVP at each point,
extracts the tail-mean NBSS slope + size-spectrum metrics. Supports
running multiple model variants (baseline, +fish, Type III, etc) in
one pass for direct comparison.

Per Benny's pref (feedback memory `feedback_scan_results`): scans are
always run fresh — no caching, no disk persistence between sessions.
Each invocation runs ~12 F_N values × ~2 variants × ~few seconds = quick.

Exposes:
- `scan_FN(model, setup, fn_values, tail_len, override_component_key)`
  Returns a dict with F_N array + per-point metrics arrays.
- `plot_slope_vs_FN(scan_results, obs_df, save_path=None)`
  Plots one or more model curves on top of the Cariaco obs envelope.
"""

import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import run_baseline_diagnostic as rbd


# =============================================================================
# CONFIG
# =============================================================================
# Cariaco obs CSV path — relative to the user's plots/ folder
DEFAULT_OBS_CSV = os.path.join('..', '..', 'data', 'processed',
                                'cariaco_monthly_euphotic_dynamic.csv')

# Reference anchors per Taniguchi_Model1_Baseline.tex §5 and Brewin 2014b
TANIGUCHI_NBSS_BASELINE = -0.993
BREWIN_GLOBAL_RANGE     = (-1.05, -0.75)


# =============================================================================
# SCAN
# =============================================================================
def scan_FN(model, setup, fn_values, phyto_esd, zoo_esd,
            tail_len=1000, override_component='Supply',
            override_param='FN', verbose=True):
    """Run a 1D F_N scan over a given model + setup.

    Parameters
    ----------
    model, setup : the XSO model + base setup (must have a parameter named
        `<override_component>__<override_param>`, default 'Supply__FN').
    fn_values : 1D array of F_N values to scan.
    phyto_esd, zoo_esd : size-class arrays.
    tail_len : trailing window for tail-mean spectrum.
    override_component, override_param : XSO setup keys for the F_N override.

    Returns
    -------
    results : dict with
        'fn_values'  : (n,)
        'N_tail'     : (n,)
        'sumP'       : (n,)
        'sumZ'       : (n,)
        'nbss_slope' : (n,)
        'centroid'   : (n,)
        'shannon'    : (n,)
        'P_bins'     : (n, 3) — Pico/Nano/Micro biomass
    """
    n = len(fn_values)
    res = {
        'fn_values':  np.array(fn_values, dtype=float),
        'N_tail':     np.full(n, np.nan),
        'sumP':       np.full(n, np.nan),
        'sumZ':       np.full(n, np.nan),
        'nbss_slope': np.full(n, np.nan),
        'centroid':   np.full(n, np.nan),
        'shannon':    np.full(n, np.nan),
        'P_bins':     np.full((n, 3), np.nan),
    }

    for k, fn in enumerate(fn_values):
        out = rbd.run_setup(
            model, setup,
            input_vars_override={override_component: {override_param: float(fn)}},
        )
        state = rbd.extract_state(out)
        summary = rbd.summarise_tail(state, tail_len=tail_len)
        metrics = rbd.compute_metrics(state, phyto_esd, zoo_esd,
                                       tail_len=tail_len)
        res['N_tail'][k]     = summary['N_tail']
        res['sumP'][k]       = summary['sumP']
        res['sumZ'][k]       = summary['sumZ']
        res['nbss_slope'][k] = metrics['nbss_slope']
        res['centroid'][k]   = metrics['centroid']
        res['shannon'][k]    = metrics['shannon']
        res['P_bins'][k, :]  = metrics['P_bins']

        if verbose:
            print(f'  F_N = {fn:5.2f}  NBSS = {metrics["nbss_slope"]:+.3f}  '
                  f'centroid = {metrics["centroid"]:+.3f}  '
                  f'ΣP = {summary["sumP"]:.3f}  ΣZ = {summary["sumZ"]:.3f}')

    return res


# =============================================================================
# OBS LOADER
# =============================================================================
def load_obs_envelope(csv_path=DEFAULT_OBS_CSV):
    """Load Cariaco per-month NBSS slope + F_N with era column."""
    df = pd.read_csv(csv_path, parse_dates=['date'])
    df = df.dropna(subset=['nbss_slope', 'FN_mmolN_m2_d']).copy()

    def _era(year):
        if year < 2005:  return 'pre-collapse'
        if year < 2014:  return 'post-collapse'
        return 'recovery'

    df['era'] = df['date'].dt.year.apply(_era)
    return df


# =============================================================================
# PLOT — model slope vs F_N over obs envelope
# =============================================================================
ERA_COLORS = {
    'pre-collapse':  '#2E86AB',
    'post-collapse': '#E07A5F',
    'recovery':      '#81B29A',
}

VARIANT_COLORS = {
    'baseline':       '#2E86AB',
    'baseline+fish':  '#C1292E',
    'Type III':       '#5B2C6F',
}


def plot_slope_vs_FN(scans_dict, obs_df=None, save_path=None,
                     title='MS3 baseline (Option A) — NBSS slope vs F_N',
                     xlim=(0, 15), ylim=(-1.4, -0.7)):
    """Plot one or more model curves on top of obs NBSS-vs-F_N envelope.

    Parameters
    ----------
    scans_dict : dict {variant_label: scan_results_dict}
        Each scan_results_dict is the return of scan_FN().
    obs_df : pd.DataFrame from load_obs_envelope(), or None to skip obs.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # ---- background: obs scatter coloured by era ----
    if obs_df is not None:
        for era, color in ERA_COLORS.items():
            sub = obs_df[obs_df['era'] == era]
            ax.scatter(sub['FN_mmolN_m2_d'], sub['nbss_slope'],
                       s=20, color=color, edgecolor='black', linewidth=0.25,
                       alpha=0.5, label=f'CARIACO obs — {era} (n={len(sub)})')

    # ---- reference anchors ----
    ax.axhline(TANIGUCHI_NBSS_BASELINE, color='dimgray', ls='--', lw=1.0,
               label=f'Taniguchi M1 analytical baseline ({TANIGUCHI_NBSS_BASELINE:.3f})')
    ax.axhspan(*BREWIN_GLOBAL_RANGE, color='gold', alpha=0.10, zorder=0,
               label=f'Brewin 2014b global range {BREWIN_GLOBAL_RANGE}')

    # ---- model curves ----
    for variant_label, scan in scans_dict.items():
        color = VARIANT_COLORS.get(variant_label, None)
        ax.plot(scan['fn_values'], scan['nbss_slope'],
                'o-', color=color, lw=2.2, ms=7, markeredgecolor='black',
                label=f'Model — {variant_label}')

    ax.set_xlabel('F_N — new nutrient supply (mmol N m⁻² d⁻¹)')
    ax.set_ylabel('NBSS slope (Platt-Denman, volume-space)')
    ax.set_title(title)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right', fontsize=8.5, framealpha=0.92)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=140, bbox_inches='tight')
        print(f'Saved figure to: {save_path}')
    return fig


# =============================================================================
# 2D SCAN — fish_rate × e_F at fixed F_N
# =============================================================================
def scan_fish_2d(model_fish, setup_fish, fish_rates, e_Fs,
                 phyto_esd, zoo_esd, fixed_FN=8.0, tail_len=1000,
                 verbose=True):
    """2D scan over fish_rate × e_F at fixed F_N.

    For each (rate, e_F) cell:
        kernel_Z(s) = (s / s_max) ** e_F             [peak=1 at largest Z]
        kernel_P(s) = 0                              [no direct phyto grazing]
        FishGrazing.rate = rate
        Supply.FN = fixed_FN

    Returns
    -------
    results : dict with
        'fish_rates'  : (n_r,)
        'e_Fs'        : (n_e,)
        'fixed_FN'    : scalar
        'nbss_slope'  : (n_r, n_e)
        'centroid'    : (n_r, n_e)
        'P_bins'      : (n_r, n_e, 3) — Pico/Nano/Micro biomass
        'sumP'        : (n_r, n_e)
        'sumZ'        : (n_r, n_e)
        'N_tail'      : (n_r, n_e)

    Note: kernel_P is held at zero throughout; only Z-side fish acts.
    Sentinel: e_F = 0 gives uniform kernel_Z = 1 (fish on all Z classes
    equally), useful contrast against size-dependent variants.
    """
    n_r = len(fish_rates)
    n_e = len(e_Fs)
    fish_rates = np.asarray(fish_rates, dtype=float)
    e_Fs       = np.asarray(e_Fs,       dtype=float)

    nbss     = np.full((n_r, n_e), np.nan)
    centroid = np.full((n_r, n_e), np.nan)
    P_bins   = np.full((n_r, n_e, 3), np.nan)
    sumP     = np.full((n_r, n_e), np.nan)
    sumZ     = np.full((n_r, n_e), np.nan)
    N_tail   = np.full((n_r, n_e), np.nan)

    kernel_P_zero = np.zeros_like(zoo_esd)
    s_max = float(zoo_esd.max())

    for i, rate in enumerate(fish_rates):
        for j, e_F in enumerate(e_Fs):
            kernel_Z = (zoo_esd / s_max) ** float(e_F)
            override = {
                'Supply':      {'FN': float(fixed_FN)},
                'FishGrazing': {'rate': float(rate),
                                'kernel_Z': kernel_Z,
                                'kernel_P': kernel_P_zero},
            }
            out = rbd.run_setup(model_fish, setup_fish,
                                input_vars_override=override)
            state = rbd.extract_state(out)
            summary = rbd.summarise_tail(state, tail_len=tail_len)
            metrics = rbd.compute_metrics(state, phyto_esd, zoo_esd,
                                           tail_len=tail_len)
            nbss[i, j]     = metrics['nbss_slope']
            centroid[i, j] = metrics['centroid']
            P_bins[i, j]   = metrics['P_bins']
            sumP[i, j]     = summary['sumP']
            sumZ[i, j]     = summary['sumZ']
            N_tail[i, j]   = summary['N_tail']

            if verbose:
                print(f'  rate={rate:5.3f}  e_F={e_F:4.2f}  '
                      f'NBSS={nbss[i,j]:+.3f}  centroid={centroid[i,j]:+.3f}  '
                      f'ΣP={sumP[i,j]:5.3f}  ΣZ={sumZ[i,j]:5.3f}')

    return dict(
        fish_rates=fish_rates, e_Fs=e_Fs, fixed_FN=float(fixed_FN),
        nbss_slope=nbss, centroid=centroid, P_bins=P_bins,
        sumP=sumP, sumZ=sumZ, N_tail=N_tail,
    )


def plot_fish_2d_heatmaps(scan_2d, baseline_nbss=None, baseline_centroid=None,
                           baseline_sumZ=None, save_path=None,
                           figsize=(13, 10)):
    """4-panel heatmap of a fish_rate × e_F 2D scan.

    Panels:
        (a) NBSS slope heatmap (RdBu_r, centred on baseline value if given)
        (b) centroid heatmap   (RdBu_r, centred on baseline value if given)
        (c) Micro biomass fraction (viridis)
        (d) ΣZ — sanity check that fish is actually suppressing zoo

    Pass `baseline_*` values (computed from the no-fish run at same F_N) to
    centre the diverging colour maps on them; otherwise use min/max ranges.
    """
    fr = scan_2d['fish_rates']
    eF = scan_2d['e_Fs']
    fixed_FN = scan_2d['fixed_FN']

    # P_bins shape (n_r, n_e, 3): Pico, Nano, Micro
    P_total = scan_2d['P_bins'].sum(axis=2)
    micro_frac = scan_2d['P_bins'][:, :, 2] / np.where(P_total > 0, P_total, np.nan)

    fig, ax = plt.subplots(2, 2, figsize=figsize)

    def _imshow(axx, data, title, cmap='viridis', vcenter=None, cbar_label=''):
        # pcolormesh on rate (y) × e_F (x). data shape (n_rates, n_e_F).
        # cell-centres are fish_rates and e_Fs; build cell-edge grids.
        def _edges(arr):
            arr = np.asarray(arr, dtype=float)
            if len(arr) == 1:
                d = 0.1
                return np.array([arr[0] - d, arr[0] + d])
            d = np.diff(arr)
            return np.concatenate([[arr[0] - d[0]/2],
                                    arr[:-1] + d/2,
                                    [arr[-1] + d[-1]/2]])
        ex = _edges(eF)
        ey = _edges(fr)
        if vcenter is not None and np.all(np.isfinite(data)):
            vmax = max(abs(np.nanmin(data) - vcenter),
                        abs(np.nanmax(data) - vcenter))
            vmin = vcenter - vmax
            vmax = vcenter + vmax
            pcm = axx.pcolormesh(ex, ey, data, cmap=cmap,
                                  vmin=vmin, vmax=vmax, shading='auto')
        else:
            pcm = axx.pcolormesh(ex, ey, data, cmap=cmap, shading='auto')
        axx.set_xlabel('e_F  (kernel_Z = (s/s_max)^e_F)')
        axx.set_ylabel('fish_rate  [d⁻¹ at peak]')
        axx.set_title(title)
        # Annotate cell values
        for i, y in enumerate(fr):
            for j, x in enumerate(eF):
                v = data[i, j]
                if np.isfinite(v):
                    axx.text(x, y, f'{v:+.2f}', ha='center', va='center',
                              fontsize=8, color='black')
        cb = plt.colorbar(pcm, ax=axx, label=cbar_label)
        if vcenter is not None:
            cb.ax.axhline(vcenter, color='black', lw=0.8)

    _imshow(ax[0, 0], scan_2d['nbss_slope'],
             '(a) NBSS slope', cmap='RdBu_r',
             vcenter=baseline_nbss, cbar_label='slope')
    _imshow(ax[0, 1], scan_2d['centroid'],
             '(b) Centroid (Sieburth)', cmap='RdBu',
             vcenter=baseline_centroid, cbar_label='centroid')
    _imshow(ax[1, 0], micro_frac,
             '(c) Micro biomass fraction', cmap='viridis',
             vcenter=None, cbar_label='Micro / ΣP')
    _imshow(ax[1, 1], scan_2d['sumZ'],
             '(d) ΣZ — fish suppression check', cmap='Reds_r',
             vcenter=baseline_sumZ, cbar_label='ΣZ [mmol N m⁻³]')

    bl_str = ''
    if baseline_nbss is not None:
        bl_str = f' (baseline-no-fish: NBSS = {baseline_nbss:+.3f})'
    plt.suptitle(f'Fish 2D scan at F_N = {fixed_FN}{bl_str}',
                  fontsize=12, y=1.005)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=140, bbox_inches='tight')
        print(f'Saved figure to: {save_path}')
    return fig


# =============================================================================
# MULTI-F_N STABILITY SCAN — variants × F_N values, max CV per cell
# =============================================================================
def scan_FN_stability(variants_dict, fn_values, phyto_esd, zoo_esd,
                       tail_len=1000, verbose=True):
    """Scan dynamical regime across F_N for multiple model variants.

    For each (variant, F_N) cell: run IVP, compute per-class CV on the tail,
    classify dynamical regime, return max CV (P) and max CV (Z).

    Parameters
    ----------
    variants_dict : dict {variant_label: (model, setup)}
        XSO model + setup pair per variant. The setup must accept an override
        ``{'Supply': {'FN': fn_value}}`` (i.e. Stock supply with FN parameter).
    fn_values : 1D array of F_N values.

    Returns
    -------
    results : dict with
        'fn_values'  : (n_FN,)
        Per variant under results['per_variant'][label]:
            'max_cv_P'  : (n_FN,)
            'max_cv_Z'  : (n_FN,)
            'regime_P'  : list of (n_FN) regime strings
            'regime_Z'  : list of (n_FN) regime strings
            'sumP'      : (n_FN,) tail-mean ΣP
            'sumZ'      : (n_FN,) tail-mean ΣZ
    """
    fn_values = np.asarray(fn_values, dtype=float)
    n_FN = len(fn_values)
    out = {
        'fn_values': fn_values,
        'per_variant': {},
    }

    for label, (model, setup) in variants_dict.items():
        if verbose:
            print(f'\n--- {label} ---')
        max_cv_P = np.full(n_FN, np.nan)
        max_cv_Z = np.full(n_FN, np.nan)
        sumP_arr = np.full(n_FN, np.nan)
        sumZ_arr = np.full(n_FN, np.nan)
        regime_P_list = []
        regime_Z_list = []

        for k, fn in enumerate(fn_values):
            outk = rbd.run_setup(
                model, setup,
                input_vars_override={'Supply': {'FN': float(fn)}},
            )
            state = rbd.extract_state(outk)
            stab  = rbd.stability_summary(state, tail_len=tail_len, verbose=False)
            summary = rbd.summarise_tail(state, tail_len=tail_len)

            max_cv_P[k] = stab['max_cv_P']
            max_cv_Z[k] = stab['max_cv_Z']
            sumP_arr[k] = summary['sumP']
            sumZ_arr[k] = summary['sumZ']
            regime_P_list.append(stab['regime_P'])
            regime_Z_list.append(stab['regime_Z'])

            if verbose:
                print(f'  F_N={fn:5.2f}  max CV_P={stab["max_cv_P"]:6.3f}  '
                      f'max CV_Z={stab["max_cv_Z"]:6.3f}  '
                      f'ΣP={summary["sumP"]:.3f}  ΣZ={summary["sumZ"]:.3f}')

        out['per_variant'][label] = dict(
            max_cv_P=max_cv_P, max_cv_Z=max_cv_Z,
            sumP=sumP_arr, sumZ=sumZ_arr,
            regime_P=regime_P_list, regime_Z=regime_Z_list,
        )

    return out


# Default colour palette for variant lines on the stability plot
STABILITY_VARIANT_COLORS = {
    'Type II':         '#2E86AB',
    'Type III':        '#C49A2E',
    'Type II + fish':  '#C1292E',
    'Type III + fish': '#8B5A00',
    'baseline':        '#2E86AB',
    'baseline+fish':   '#C1292E',
}


def plot_stability_vs_FN(stab_results, save_path=None, figsize=(12, 8),
                          y_log=True):
    """Max-CV vs F_N curves for each variant; P and Z in two panels.

    Reference lines at CV=0.05 (fixed-point threshold) and CV=0.5
    (large-amplitude threshold) per the May 22 2026 bridge-figure
    convention used in run_baseline_diagnostic.regime_from_cv.
    """
    fn = stab_results['fn_values']
    per = stab_results['per_variant']

    fig, ax = plt.subplots(2, 1, figsize=figsize, sharex=True)

    for label, data in per.items():
        color = STABILITY_VARIANT_COLORS.get(label, None)
        ax[0].plot(fn, data['max_cv_P'], 'o-', color=color,
                    lw=2.0, ms=7, markeredgecolor='black', label=label)
        ax[1].plot(fn, data['max_cv_Z'], 'o-', color=color,
                    lw=2.0, ms=7, markeredgecolor='black', label=label)

    for a, side in zip(ax, ['P', 'Z']):
        a.axhline(0.05, color='red',    ls='--', lw=0.9,
                   label='CV=0.05 (fixed-point ceiling)')
        a.axhline(0.50, color='maroon', ls=':',  lw=0.9,
                   label='CV=0.50 (large-amplitude floor)')
        a.set_ylabel(f'max CV of {side}_i over tail')
        if y_log:
            a.set_yscale('log')
        a.grid(alpha=0.3, which='both')
        a.legend(loc='best', fontsize=8.5, framealpha=0.92)

    ax[0].set_title('(a) Phyto stability — max CV across active classes')
    ax[1].set_title('(b) Zoo stability — max CV across active classes')
    ax[-1].set_xlabel('F_N — new nutrient supply [mmol N m⁻² d⁻¹]')

    plt.suptitle('Dynamical stability regime vs nutrient forcing',
                  fontsize=12, y=1.005)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=140, bbox_inches='tight')
        print(f'Saved figure to: {save_path}')
    return fig


# =============================================================================
# MAIN — default baseline + fish scan over the Cariaco F_N range
# =============================================================================
def main(fn_values=None, save_path='fig_slope_vs_FN.png',
         csv_path=DEFAULT_OBS_CSV):
    """End-to-end: scan baseline + fish over the obs F_N range, plot."""
    import cariaco_baseline_setups as cbs

    if fn_values is None:
        fn_values = np.linspace(0.5, 15.0, 12)

    print('--- Baseline (Taniguchi M1 biology + Stock supply) ---')
    scan_base = scan_FN(cbs.model_baseline, cbs.model_setup_baseline,
                        fn_values, cbs.phyto_esd, cbs.zoo_esd)

    print('\n--- Baseline + simplified fish (power-law kernel on Z) ---')
    scan_fish = scan_FN(cbs.model_baseline_fish, cbs.model_setup_baseline_fish,
                        fn_values, cbs.phyto_esd, cbs.zoo_esd)

    obs_df = load_obs_envelope(csv_path)
    fig = plot_slope_vs_FN(
        {'baseline': scan_base, 'baseline+fish': scan_fish},
        obs_df=obs_df, save_path=save_path,
    )
    plt.show()
    return {'baseline': scan_base, 'baseline+fish': scan_fish}, fig


if __name__ == '__main__':
    main()
