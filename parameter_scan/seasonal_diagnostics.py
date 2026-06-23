"""
seasonal_diagnostics.py -- model-vs-obs diagnostics for the seasonal harness.

Analysis-over-output layer for `seasonal_scan_harness`: it takes ONE run's stored
output -- the climatology summary `clim` and (optionally) the full trajectory `r`
from `run_one(..., return_traj=True)` -- and compares it to the per-era obs, both
as a printed table (every metric, both statistics, + flags) and as three plot views
matched to each metric's character:

  - cycles  : model per-calendar-month climatology over the obs monthly cloud
              (mcs / composition / Zoo>200 -- where bloom TIMING is the point)
  - boxes   : model post-spin-up distribution vs obs monthly distribution
              (Export / PP / N / sumP / Zoo>200 / Zoo>500 -- bulk + flux, where the
              distribution is the point and timing less so)
  - fn_loop : the within-year (F_N, mcs) loop over the obs (F_N, mcs) cloud

Nothing is scored here that the harness doesn't already define: the print reuses
`ssh.score` and the obs fingerprint. Flags ANNOTATE, never filter. Built to be
driven from a parameter_scan/FINAL notebook; the run and obs are kept general so
you re-analyse without re-running.

Companion to seasonal_scan_harness.py; lives beside it on the import path.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seasonal_scan_harness as ssh

# Everything we can compare; trim the headline score later (a scan optimising too
# few points can never hit all obs -- see the print's per-metric distances).
PRINT_METRICS = ['mcs', 'pico', 'nano', 'micro', 'sumP', 'N', 'PP', 'Export', 'Z200', 'Z500']
CYCLE_METRICS = ('mcs', 'micro', 'Z200')                       # timing matters -> cycle view
BOX_METRICS = ('Export', 'PP', 'N', 'sumP', 'Z200', 'Z500')   # bulk/flux -> distribution view
_MO = np.arange(1, 13)
MOD_C, OBS_C = '#C0563F', '0.55'
LABELS = {'mcs': 'mean cell size (um)', 'pico': 'Pico frac', 'nano': 'Nano frac',
          'micro': 'Micro frac', 'sumP': 'sumP (mmolN)', 'N': 'N (mmolN)',
          'PP': 'PP (mmolN/m3/d)', 'Export': 'Export (mmolN/m2/d)',
          'Z200': 'Zoo>200 (mmolN)', 'Z500': 'Zoo>500 (mmolN)'}


# ----------------------------------------------------------------------------- helpers
def _post_spinup(r, key, spinup):
    """Trajectory values of `key` after the spin-up window, finite only."""
    if r is None or key not in r:
        return np.array([])
    keep = r['t'] >= spinup * 365.0
    v = np.asarray(r[key])[keep]
    return v[np.isfinite(v)]


def _obs_month_median(obs_m, key):
    """Obs per-calendar-month median of `key` (12 values, NaN where no obs that month)."""
    s = obs_m.dropna(subset=[key])
    return s.groupby('mo')[key].median().reindex(_MO).to_numpy()


def _peak_month(cycle):
    """Calendar month (1-12) of the max of a 12-vector, ignoring NaN."""
    cycle = np.asarray(cycle, float)
    if not np.any(np.isfinite(cycle)):
        return np.nan
    return int(np.nanargmax(cycle) + 1)


def _model_monthly(r, key, spinup):
    """Post-spin-up per-calendar-month mean of trajectory `key` (12 values), using the
    same month assignment as ssh._clim. NaN-safe (no empty-slice warning on blow-ups)."""
    keep = r['t'] >= spinup * 365.0
    mo = np.clip(np.searchsorted(ssh._MEDGES, np.mod(r['t'][keep], 365.0), 'right') + 1, 1, 12)
    v = np.asarray(r[key], float)[keep]
    out = np.full(12, np.nan)
    for i, m in enumerate(_MO):
        sel = v[mo == m]
        sel = sel[np.isfinite(sel)]
        if sel.size:
            out[i] = sel.mean()
    return out


# ----------------------------------------------------------------------------- print
def diagnostic_print(clim, group, obs_targets, obs_monthly, metrics=PRINT_METRICS):
    """Compact model-vs-obs table (both stats) + scores + flags for one run.
    Returns a dict of the headline numbers."""
    om = obs_targets[group]                                    # {'med':..., 'mean':...}
    obs_m = obs_monthly[group]
    nan = bool(clim.get('has_nan', False))
    flag = "   *** has_nan -- long-run stats NULLED ***" if nan else ""
    print(f"=== diagnose [{group}] ==={flag}")
    print(f"{'metric':8s} {'mod_mean':>9s} {'obs_mean':>9s} {'mod_med':>9s} {'obs_med':>9s} {'d_mean':>7s} {'d_med':>7s}")
    for k in metrics:
        mm, mo = clim.get(k, np.nan), om['mean'].get(k, np.nan)
        cm, co = clim.get(k + '_med', np.nan), om['med'].get(k, np.nan)
        d_mean = (mm - mo) / mo if (np.isfinite(mo) and abs(mo) > 1e-12) else np.nan
        d_med = (cm - co) / co if (np.isfinite(co) and abs(co) > 1e-12) else np.nan
        print(f"{k:8s} {mm:9.3g} {mo:9.3g} {cm:9.3g} {co:9.3g} {d_mean:+7.2f} {d_med:+7.2f}")

    sc_mean = ssh.score({k: clim.get(k, np.nan) for k in ssh.SCORE_KEYS}, om['mean'])
    sc_med = ssh.score({k: clim.get(k + '_med', np.nan) for k in ssh.SCORE_KEYS}, om['med'])
    _finite = [v for v in (sc_mean, sc_med) if np.isfinite(v)]
    joint = float(max(_finite)) if _finite else np.nan       # NaN if the run is blown up
    pk_mod = _peak_month(clim.get('clim_mcs', np.full(12, np.nan)))
    pk_obs = _peak_month(_obs_month_median(obs_m, 'mcs'))
    off = (pk_mod - pk_obs) if (np.isfinite(pk_mod) and np.isfinite(pk_obs)) else np.nan

    print(f"score   mean={sc_mean:.3f}  med={sc_med:.3f}  joint(max)={joint:.3f}   (on {ssh.SCORE_KEYS})")
    print(f"flags   has_nan={nan} nan_frac={clim.get('nan_frac', 0.0):.2f} "
          f"mcs_conv={clim.get('mcs_conv', np.nan):.2f} Z200_peak={clim.get('Z200_peak', np.nan):.3f} "
          f"cv_sumP={clim.get('cv_sumP', np.nan):.2f}")
    print(f"        mcs_peak month  model={pk_mod}  obs={pk_obs}  offset={off}")
    return dict(score_mean=sc_mean, score_med=sc_med, joint=joint, peak_offset=off)


# ----------------------------------------------------------------------------- plots
def plot_cycles(clim, group, obs_monthly, metrics=CYCLE_METRICS, axes=None):
    """Model per-month climatology (line) over the obs monthly cloud + obs monthly median."""
    obs_m = obs_monthly[group]
    if axes is None:
        fig, axes = plt.subplots(1, len(metrics), figsize=(3.6 * len(metrics), 3.0))
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, metrics):
        s = obs_m.dropna(subset=[k])
        ax.scatter(s['mo'], s[k], s=16, color=OBS_C, alpha=0.45, lw=0, zorder=2, label='obs')
        ax.plot(_MO, _obs_month_median(obs_m, k), 'o-', color='0.25', lw=1.3, ms=3, zorder=3, label='obs median')
        ck = clim.get('clim_' + k)
        if ck is not None:
            ax.plot(_MO, ck, 's-', color=MOD_C, lw=1.8, ms=3.5, zorder=4, label='model')
        ax.set_xticks(_MO); ax.set_xlabel('month'); ax.set_title(LABELS.get(k, k), fontsize=9)
        ax.set_ylim(bottom=0)
    axes[0].legend(fontsize=7, frameon=False)
    fig = axes[0].figure
    fig.suptitle(f"cycles [{group}] -- model vs obs monthly", fontsize=10)
    fig.tight_layout()
    return fig


def plot_boxes(r, group, obs_monthly, spinup, metrics=BOX_METRICS, axes=None):
    """Model post-spin-up distribution vs obs monthly distribution, per metric."""
    obs_m = obs_monthly[group]
    if axes is None:
        fig, axes = plt.subplots(1, len(metrics), figsize=(2.1 * len(metrics), 3.2))
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, metrics):
        mod = _post_spinup(r, k, spinup)
        obs = obs_m[k].dropna().to_numpy() if k in obs_m else np.array([])
        data, labs, cols = [], [], []
        if len(mod):
            data.append(mod); labs.append('model'); cols.append(MOD_C)
        if len(obs):
            data.append(obs); labs.append('obs'); cols.append(OBS_C)
        if data:
            bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=False,
                            medianprops=dict(color='k', lw=1.1))
            for b, c in zip(bp['boxes'], cols):
                b.set_facecolor(c); b.set_alpha(0.5)
            ax.set_xticks(range(1, len(labs) + 1)); ax.set_xticklabels(labs, fontsize=8)
        else:
            ax.text(0.5, 0.5, '[no data]', ha='center', va='center', transform=ax.transAxes, fontsize=8)
        ax.set_title(LABELS.get(k, k), fontsize=8.5); ax.set_ylim(bottom=0); ax.grid(alpha=0.2, axis='y')
    fig = axes[0].figure
    fig.suptitle(f"distributions [{group}] -- model (post-spin-up) vs obs monthly", fontsize=10)
    fig.tight_layout()
    return fig


def plot_fn_loop(r, group, obs_monthly, spinup, ax=None):
    """Within-year (F_N, mcs) loop: model monthly-clim path over the obs (F_N, mcs) cloud."""
    obs_m = obs_monthly[group]
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.2, 3.8))
    s = obs_m.dropna(subset=['FN', 'mcs'])
    ax.scatter(s['FN'], s['mcs'], s=18, color=OBS_C, alpha=0.45, lw=0, zorder=2, label='obs')
    fn_c, mcs_c = _model_monthly(r, 'FN', spinup), _model_monthly(r, 'mcs', spinup)
    finite = np.isfinite(fn_c) & np.isfinite(mcs_c)
    if finite.any():
        loop = np.r_[np.arange(12), 0]                         # close Jan -> Dec -> Jan
        ax.plot(fn_c[loop], mcs_c[loop], 'o-', color=MOD_C, lw=1.6, ms=4, zorder=4, label='model (monthly)')
        for m in _MO:
            if finite[m - 1]:                                  # skip non-finite (blown-up) months
                ax.annotate(str(m), (fn_c[m - 1], mcs_c[m - 1]), fontsize=6, color='#7a2f1f', zorder=5)
    else:
        ax.text(0.5, 0.95, 'model: no finite data (run NaN-terminated)', ha='center', va='top',
                transform=ax.transAxes, fontsize=8, color=MOD_C)
    ax.set_xlabel('F_N (mmolN/m2/d)'); ax.set_ylabel('mean cell size (um)')
    ax.set_title(f"F_N - mcs loop [{group}]", fontsize=9); ax.legend(fontsize=7, frameon=False)
    ax.figure.tight_layout()
    return ax.figure


def diagnose(clim, r, group, spinup, obs_targets, obs_monthly, title=''):
    """Full per-run diagnostic: print table + flags, then the cycles / boxes / loop figures.
    Pass r=None to print + cycles only (summary-grain run)."""
    if title:
        print(title)
    summ = diagnostic_print(clim, group, obs_targets, obs_monthly)
    if clim.get('has_nan', False):
        print("  (run NaN-terminated -- plots show what finite data remain)")
    plot_cycles(clim, group, obs_monthly)
    if r is not None:
        plot_boxes(r, group, obs_monthly, spinup)
        plot_fn_loop(r, group, obs_monthly, spinup)
    plt.show()
    return summ
