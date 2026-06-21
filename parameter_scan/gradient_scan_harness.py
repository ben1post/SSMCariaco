"""
gradient_scan_harness.py — reusable harness for MS3 gradient parameter scans.

Built on parscan_utils_extended (flags + extraction + panel plots) and
xso.parscans.run_xso_parscan. Construct-agnostic: a CONSTRUCT spec (model/setup +
fixed allometry/params) and an AXES dict (param -> values) fully define a scan, so
switching allometries or sweep parameters is a declaration change, not a rewrite.

Design (settled with Benny, 2026-06-21):
- Forcing is the empirical F_N -> d_e -> T gradient via run_xso_parscan
  `linked_overrides` (passed in by the driver; not hardcoded here).
- Flags ANNOTATE, never filter (parscan_utils_extended philosophy). The harness
  ranks and reports; nothing is excluded.
- Fit = curve-on-curve normalised-RMS distance to the F_N-resolved obs curves over
  F_N <= fn_score (the obs-dense range), across mcs + Pico/Nano/Micro fractions +
  SumP + N.
- Penalty is multiplicative and itemised: combined = fit * (1 + penalty), with
  extinction (collapse / coexistence-loss) weighted high and every other flag at a
  single 'other' weight. Tunable, and re-rankable WITHOUT re-running
  (ScanResult.reweight).
- Three views on one scan: 'combined' (fit with penalty), 'clean' (flag-free subset
  ranked by pure fit), 'best_fit' (global fit minimum, flags ignored for order).
- Per-candidate inspection: ScanResult.show / show_top (full 8-panel
  plot_scan_panels_1d) and .overview (small-multiples mcs(F_N) vs obs per category).
- Cross-construct comparison: ScanRegistry merges ScanResults (identical scoring)
  into combined / clean / best views spanning every construct tried this session.

NOTE (Benny's norm): every scan re-runs fresh — the registry is SESSION-scoped and
the saved .nc / .png are session artifacts, not a cache.

Solver settings are always passed in explicitly (never baked in here), per the
"surface solver settings every time" rule.
"""

import io
import json
import time
import contextlib
import itertools

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import parscan_utils_extended as pue
from xso.parscans import run_xso_parscan


# =============================================================================
# Obs side — F_N-resolved curves (single source of truth for scoring)
# =============================================================================
def _fn_binned(fn, v, n_bins=6, min_per_bin=3):
    """Quantile-bin v by fn; return (bin-centre fn, median v). Same convention as
    pue.build_obs_fraction_curve. Empty arrays if too few finite points."""
    fn = np.asarray(fn, float)
    v = np.asarray(v, float)
    ok = np.isfinite(fn) & np.isfinite(v)
    fn, v = fn[ok], v[ok]
    if fn.size < min_per_bin:
        return np.array([]), np.array([])
    edges = np.quantile(fn, np.linspace(0.0, 1.0, n_bins + 1))
    fc, vc = [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (fn >= lo) & (fn <= hi) if i == n_bins - 1 else (fn >= lo) & (fn < hi)
        if int(m.sum()) >= min_per_bin:
            fc.append(float(np.median(fn[m])))
            vc.append(float(np.median(v[m])))
    o = np.argsort(fc)
    return np.array(fc)[o], np.array(vc)[o]


def build_obs_curves(monthly_df, fn_col='FN_mmolN_m2_d',
                     pico='pico_mmolN', nano='nano_mmolN', micro='micro_mmolN',
                     n_col='NO3_mmolN', n_bins=6, min_per_bin=3, bin_geomeans=None):
    """F_N-resolved obs medians for the scored targets.

    Returns a dict:
      'frac' : {'fn','pico','nano','micro'} from pue.build_obs_fraction_curve
      'mcs'  : (fn, value)   per-month mcs (fractions x geomeans), then F_N-binned
      'sumP' : (fn, value)   per-month Pico+Nano+Micro, F_N-binned
      'N'    : (fn, value)   per-month NO3, F_N-binned
    mcs is computed per-month then binned, consistent with build_obs_refs."""
    if bin_geomeans is None:
        bin_geomeans = pue.CARIACO_PHYTO_BIN_GEOMEANS
    geom = np.asarray(bin_geomeans, float)
    fn = np.asarray(monthly_df[fn_col], float)
    P = monthly_df[[pico, nano, micro]].to_numpy(float)
    tot = P.sum(axis=1)
    frac = P / np.where(tot[:, None] > 0, tot[:, None], np.nan)
    mcs_month = 10.0 ** (frac @ np.log10(geom))

    ofc = pue.build_obs_fraction_curve(monthly_df, fn_col=fn_col, pico_col=pico,
                                       nano_col=nano, micro_col=micro,
                                       n_bins=n_bins, min_per_bin=min_per_bin)
    if ofc is None:
        ofc = {'fn': np.array([]), 'pico': np.array([]),
               'nano': np.array([]), 'micro': np.array([])}
    return {
        'frac': ofc,
        'mcs':  _fn_binned(fn, mcs_month, n_bins, min_per_bin),
        'sumP': _fn_binned(fn, tot, n_bins, min_per_bin),
        'N':    _fn_binned(fn, monthly_df[n_col].to_numpy(float), n_bins, min_per_bin),
    }


# =============================================================================
# Fit (curve-on-curve) + penalty (itemised, multiplicative)
# =============================================================================
DEFAULT_SCORE_TARGETS = ('mcs', 'fpico', 'fnano', 'fmicro', 'sumP', 'N')
DEFAULT_PENALTY_WEIGHTS = {'extinction': 5.0, 'other': 0.5}

# The flag conditions that make up the penalty, in display order. Extinction is
# weighted separately ('extinction'); everything else shares the single 'other'.
_OTHER_TERMS = ['unstable', 'n_runaway', 'descending',
                'microA_fail', 'nanoB_fail', 'micro_short']
_TERM_LABEL = {'unstable': 'unstable', 'n_runaway': 'N-runaway',
               'descending': 'descending', 'microA_fail': 'microA-fail',
               'nanoB_fail': 'nanoB-fail', 'micro_short': 'micro-short'}


def cc_score(metrics, obs_curves, fn_score=5.0, weights=None,
             targets=DEFAULT_SCORE_TARGETS):
    """Normalised-RMS curve-on-curve distance to obs over F_N <= fn_score.

    Per target t and model point i (with F_N_i <= fn_score):
        d_ti = (model_t,i - obs_t(F_N_i)) / obs_t(F_N_i)
    score = sqrt( sum_t w_t * mean_i d_ti^2 / sum_t w_t ).
    `weights` is an optional {target: w} dict (default 1.0 each) so ΣP / N can be
    down-weighted vs composition if they dominate."""
    ax = np.asarray(metrics['axis'], float)
    sel = ax <= fn_score
    sp = np.asarray(metrics['sumP'], float)
    spd = np.where(sp > 0, sp, np.nan)
    ofc = obs_curves['frac']
    model_obs = {
        'mcs':    (np.asarray(metrics['mcs'], float),         obs_curves['mcs']),
        'fpico':  (np.asarray(metrics['P_pico'], float) / spd, (ofc['fn'], ofc['pico'])),
        'fnano':  (np.asarray(metrics['P_nano'], float) / spd, (ofc['fn'], ofc['nano'])),
        'fmicro': (np.asarray(metrics['P_micro'], float) / spd, (ofc['fn'], ofc['micro'])),
        'sumP':   (sp,                                        obs_curves['sumP']),
        'N':      (np.asarray(metrics['N'], float),           obs_curves['N']),
    }
    if weights is None:
        weights = {t: 1.0 for t in targets}
    num, wsum = 0.0, 0.0
    for t in targets:
        mod, (ofn, oval) = model_obs[t]
        if np.size(ofn) < 2:
            continue
        obs = np.interp(ax, ofn, oval)
        d = (mod - obs) / np.where(np.abs(obs) > 1e-12, obs, np.nan)
        d = d[sel]
        d = d[np.isfinite(d)]
        if d.size == 0:
            continue
        w = weights.get(t, 1.0)
        num += w * float(np.mean(d ** 2))
        wsum += w
    return float(np.sqrt(num / wsum)) if wsum > 0 else np.nan


def flag_penalty(flag_state, weights=None):
    """Itemised multiplicative penalty from a windowed flag-state dict.

    Returns (penalty, fired) where `fired` lists the contributing terms.
    combined = fit * (1 + penalty). Extinction (collapse OR coexistence-loss in
    the gated window) is weighted at weights['extinction']; every other condition
    (unstable, N-runaway, descending, Flag A fail, Flag B fail, micro-short) at the
    single weights['other']. A clean run has fired == []  ->  penalty 0."""
    w = {**DEFAULT_PENALTY_WEIGHTS, **(weights or {})}
    fired = []
    pen = 0.0
    if flag_state.get('extinction', False):
        pen += w['extinction']
        fired.append('extinction')
    others = [_TERM_LABEL[k] for k in _OTHER_TERMS if flag_state.get(k, False)]
    pen += w['other'] * len(others)
    fired += others
    return pen, fired


def _flag_state(flags, win):
    """Compact windowed flag-state from a compute_flags result (per-point arrays
    AND-ed with the in-window mask; per-curve gates already windowed by
    fn_eval_max)."""
    def _any(key):
        return bool(np.any(np.asarray(flags[key])[win]))
    return {
        'extinction':  _any('collapse') or _any('coexist_loss'),
        'unstable':    _any('unstable') or _any('strong_cycle'),
        'n_runaway':   _any('n_runaway'),
        'descending':  bool(flags['mcs_descending']),
        'microA_fail': flags['micro_ok_highFN'] is False,
        'nanoB_fail':  flags['nano_trough_ok'] is False,
        'micro_short': _any('micro_short'),
        'admissible':  bool(flags['admissible_in_window']),
    }


# =============================================================================
# ScanResult — one construct's scan: records, metrics, curves + the views
# =============================================================================
class ScanResult:
    def __init__(self, name, records, metrics, curves_ds, obs_curves, obs_refs,
                 fn_values, fn_gate, fn_score, penalty_weights):
        self.name = name
        self.records = records                 # list of per-combo dicts
        self.metrics = metrics                 # label -> metrics dict (None if failed)
        self.curves = curves_ds                # xr.Dataset (combo x F_N)
        self.obs_curves = obs_curves
        self.obs_refs = obs_refs
        self.fn_values = np.asarray(fn_values, float)
        self.fn_gate = fn_gate
        self.fn_score = fn_score
        self.reweight(penalty_weights)

    # -- ranking ----------------------------------------------------------------
    def reweight(self, penalty_weights):
        """Recompute penalty + combined for every record (no re-run)."""
        self.penalty_weights = {**DEFAULT_PENALTY_WEIGHTS, **(penalty_weights or {})}
        for r in self.records:
            pen, fired = flag_penalty(r['fs'], self.penalty_weights)
            r['penalty'] = pen
            r['fired'] = fired
            r['clean'] = (len(fired) == 0)
            r['combined'] = r['fit'] * (1.0 + pen) if np.isfinite(r['fit']) else np.inf
        return self

    def ranked(self, view='combined'):
        if view == 'combined':
            key = lambda r: r['combined'] if np.isfinite(r['combined']) else np.inf
            return sorted(self.records, key=key)
        if view == 'clean':
            key = lambda r: r['fit'] if np.isfinite(r['fit']) else np.inf
            return sorted([r for r in self.records if r['clean']], key=key)
        if view == 'best_fit':
            key = lambda r: r['fit'] if np.isfinite(r['fit']) else np.inf
            return sorted(self.records, key=key)
        raise ValueError(f"view must be 'combined' | 'clean' | 'best_fit', got {view!r}")

    # -- tables -----------------------------------------------------------------
    def print_table(self, view='combined', n=20):
        rk = self.ranked(view)
        o = self.obs_refs
        w = self.penalty_weights
        print(f"\n=== [{self.name}] view='{view}' — top {min(n, len(rk))} of "
              f"{len(self.records)} (ext={w['extinction']:g}, other={w['other']:g}) ===")
        print(f"obs: mcs={o['mcs']:.2f}  N={o['N']:.2f}  ΣP={o['sumP']:.2f}")
        if not rk:
            print("  (none)")
            return
        print(f"{'params':32s}{'fit':>7}{'pen':>6}{'comb':>7}{'mcs_hi':>7}"
              f"{'P/N/M@gate':>17}{'maxN':>7}{'maxΣP':>7}  flags")
        for r in rk[:n]:
            pnm = f"{r['pico_top']:.2f}/{r['nano_top']:.2f}/{r['micro_top']:.2f}"
            fit = f"{r['fit']:7.3f}" if np.isfinite(r['fit']) else f"{'-':>7}"
            comb = f"{r['combined']:7.3f}" if np.isfinite(r['combined']) else f"{'inf':>7}"
            print(f"{r['label']:32s}{fit}{r['penalty']:6.2f}{comb}{r['mcs_hi']:7.2f}"
                  f"{pnm:>17}{r['maxN']:7.2f}{r['maxSumP']:7.2f}  "
                  f"{','.join(r['fired']) if r['fired'] else 'clean'}")

    def print_all(self, n=20):
        self.print_table('combined', n)
        self.print_table('clean', n)
        self.print_table('best_fit', max(5, n // 4))

    # -- inspection -------------------------------------------------------------
    def show(self, label):
        """Full 8-panel diagnostic (plot_scan_panels_1d) for one combo."""
        m = self.metrics.get(label)
        if m is None:
            print(f"{label}: failed run — no metrics")
            return None
        return pue.plot_scan_panels_1d(
            m, self.obs_refs, header=f"[{self.name}] {label}",
            axis_label='F_N', obs_frac_curve=self.obs_curves['frac'])

    def show_top(self, view='combined', k=3):
        return [self.show(r['label']) for r in self.ranked(view)[:k]]

    def overview(self, view='combined', k=6, ncol=3):
        """Small-multiples quick look: mcs(F_N) vs obs for the top-k of a view,
        title carrying fit / penalty / composition-at-gate / flags."""
        rk = self.ranked(view)[:k]
        if not rk:
            print(f"[{self.name}] view='{view}': nothing to show")
            return None
        nrow = int(np.ceil(len(rk) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.2 * nrow),
                                 squeeze=False)
        omcs_fn, omcs_v = self.obs_curves['mcs']
        for ax, r in zip(axes.flat, rk):
            ax.plot(self.fn_values, self.curves['mcs'].sel(combo=r['label']).values,
                    '-', color='C0', label='model')
            if np.size(omcs_fn) >= 2:
                ax.plot(omcs_fn, omcs_v, 'k--o', ms=3, label='obs')
            ax.axvline(self.fn_score, color='grey', ls=':', lw=1)
            ax.axvline(self.fn_gate, color='grey', ls=':', lw=1)
            pnm = f"{r['pico_top']:.2f}/{r['nano_top']:.2f}/{r['micro_top']:.2f}"
            fl = ','.join(r['fired']) if r['fired'] else 'clean'
            fit = r['fit'] if np.isfinite(r['fit']) else np.nan
            ax.set_title(f"{r['label']}\nfit={fit:.2f} pen={r['penalty']:.2f} "
                         f"P/N/M={pnm}\n[{fl}]", fontsize=8)
            ax.set_xlabel('F_N'); ax.set_ylabel('mcs [µm]')
        for ax in axes.flat[len(rk):]:
            ax.axis('off')
        fig.suptitle(f"[{self.name}] overview — view='{view}'  "
                     f"(grey dotted = score≤{self.fn_score:g} / gate≤{self.fn_gate:g})")
        plt.tight_layout()
        return fig


# =============================================================================
# Engine — run a construct over the product of axes, along the linked gradient
# =============================================================================
DEFAULT_CURVE_KEYS = ['mcs', 'P_pico', 'P_nano', 'P_micro', 'sumP', 'sumZ',
                      'Z_gt200', 'Z_gt500', 'N', 'D', 'PP', 'Export',
                      'cv_sumP', 'tail_has_nan']


def run_gradient_scan(name, construct, axes, obs_curves, obs_refs, fn_values, linked,
                      solver_kwargs, *, fn_gate=8.0, fn_score=5.0,
                      score_weights=None, penalty_weights=None, processes=8,
                      avg_window=1000, nc_path=None, progress_every=14,
                      curve_keys=None):
    """Run `construct` over the Cartesian product of `axes` along the linked F_N
    gradient and return a ScanResult.

    construct : {'model_file_name', 'model_name', 'model_setup_name',
                 'fixed_overrides'} — fixed_overrides carries the construct's fixed
                 allometry arrays / scalars.
    axes      : {param_name: 1d array} — the product gives the combos; each combo's
                values override construct['fixed_overrides'].
    linked    : {'Inflow__de': arr, 'Temperature__value': arr}, len == len(fn_values).
    solver_kwargs : dict, serialized to Core__solver_kwargs (surfaced, not baked in).
    """
    if curve_keys is None:
        curve_keys = list(DEFAULT_CURVE_KEYS)
    fn_values = np.asarray(fn_values, float)
    de_med = float(np.median(np.asarray(linked['Inflow__de'], float)))
    solver_json = json.dumps(solver_kwargs)
    ax_names = list(axes.keys())
    combos = list(itertools.product(*[np.asarray(axes[k], float) for k in ax_names]))
    n_total = len(combos)
    print(f"[{name}] {n_total} combos over {ax_names}, {len(fn_values)} F_N pts "
          f"| gate≤{fn_gate:g}  score≤{fn_score:g}")

    records, metrics, curves = [], {}, {}
    t0, sink = time.time(), io.StringIO()
    for i, vals in enumerate(combos):
        combo = {k: float(v) for k, v in zip(ax_names, vals)}
        label = '|'.join(f"{k.split('__')[-1]}={v:g}" for k, v in combo.items())
        fixed = {**construct['fixed_overrides'], **combo,
                 'Core__solver_kwargs': solver_json}
        try:
            with contextlib.redirect_stdout(sink):
                scan = run_xso_parscan(
                    model_file_name=construct['model_file_name'],
                    model_name=construct['model_name'],
                    model_setup_name=construct['model_setup_name'],
                    param_name='Inflow__FN', param_values=fn_values,
                    linked_overrides=linked, fixed_overrides=fixed,
                    postprocess_name='avg_tail_stats',
                    postprocess_kwargs={'avg_window': avg_window}, processes=processes)
            m = pue.extract_scan_metrics_1d(scan, 'Inflow__FN', {'Inflow__de': de_med})
            f = pue.compute_flags(m, obs_refs, obs_frac_curve=obs_curves['frac'],
                                  fn_eval_max=fn_gate)
            ax = np.asarray(m['axis'], float)
            win = ax <= fn_gate
            fs = _flag_state(f, win)
            top = np.where(win)[0][int(np.argmax(ax[win]))]
            spt = m['sumP'][top]
            records.append(dict(
                label=label, params=combo,
                fit=cc_score(m, obs_curves, fn_score, score_weights),
                fs=fs, microA=f['micro_ok_highFN'], nanoB=f['nano_trough_ok'],
                mcs_hi=float(np.nanmax(m['mcs'])),
                pico_top=float(m['P_pico'][top] / spt) if spt > 0 else np.nan,
                nano_top=float(m['P_nano'][top] / spt) if spt > 0 else np.nan,
                micro_top=float(m['P_micro'][top] / spt) if spt > 0 else np.nan,
                maxN=float(np.nanmax(np.where(win, m['N'], np.nan))),
                maxSumP=float(np.nanmax(np.where(win, m['sumP'], np.nan)))))
            metrics[label] = m
            curves[label] = {k: np.asarray(m[k], float) for k in curve_keys if k in m}
        except Exception as e:
            fs = {k: False for k in ('extinction', 'unstable', 'n_runaway',
                                     'descending', 'microA_fail', 'nanoB_fail',
                                     'micro_short', 'admissible')}
            fs['extinction'] = True   # failed run ranks with the extinction class
            records.append(dict(label=label, params=combo, fit=np.inf, fs=fs,
                                microA=None, nanoB=None, mcs_hi=np.nan,
                                pico_top=np.nan, nano_top=np.nan, micro_top=np.nan,
                                maxN=np.nan, maxSumP=np.nan))
            metrics[label] = None
            curves[label] = {k: np.full(len(fn_values), np.nan) for k in curve_keys}
            print(f"  [warn] {label}: {e}")
        if progress_every and ((i + 1) % progress_every == 0 or (i + 1) == n_total):
            el = time.time() - t0
            print(f"  {i+1:3d}/{n_total} | {el/60:4.1f} min | "
                  f"ETA {el/(i+1)*(n_total-i-1)/60:4.1f} min")

    cl = list(curves.keys())
    curves_ds = xr.Dataset(
        {k: (('combo', 'F_N'),
             np.vstack([curves[c].get(k, np.full(len(fn_values), np.nan)) for c in cl]))
         for k in curve_keys},
        coords={'combo': cl, 'F_N': fn_values})
    if nc_path:
        curves_ds.to_netcdf(nc_path)
        print(f"[{name}] saved curves -> {nc_path}")

    return ScanResult(name, records, metrics, curves_ds, obs_curves, obs_refs,
                      fn_values, fn_gate, fn_score, penalty_weights)


# =============================================================================
# ScanRegistry — compare candidates ACROSS constructs (session-scoped)
# =============================================================================
class ScanRegistry:
    def __init__(self):
        self.results = {}                      # construct name -> ScanResult

    def add(self, result):
        self.results[result.name] = result
        return self

    def reweight(self, penalty_weights):
        for res in self.results.values():
            res.reweight(penalty_weights)
        return self

    def _all(self):
        out = []
        for name, res in self.results.items():
            for r in res.records:
                rr = dict(r)
                rr['construct'] = name
                out.append(rr)
        return out

    def ranked(self, view='combined'):
        recs = self._all()
        if view == 'combined':
            return sorted(recs, key=lambda r: r['combined'] if np.isfinite(r['combined']) else np.inf)
        if view == 'clean':
            recs = [r for r in recs if r['clean']]
            return sorted(recs, key=lambda r: r['fit'] if np.isfinite(r['fit']) else np.inf)
        if view == 'best_fit':
            return sorted(recs, key=lambda r: r['fit'] if np.isfinite(r['fit']) else np.inf)
        raise ValueError(f"view must be 'combined' | 'clean' | 'best_fit', got {view!r}")

    def print_table(self, view='combined', n=25):
        rk = self.ranked(view)
        print(f"\n=== REGISTRY view='{view}' — top {min(n, len(rk))} across "
              f"{len(self.results)} constructs ===")
        if not rk:
            print("  (none)")
            return
        print(f"{'construct':16s}{'params':32s}{'fit':>7}{'pen':>6}{'comb':>7}"
              f"{'mcs_hi':>7}  flags")
        for r in rk[:n]:
            fit = f"{r['fit']:7.3f}" if np.isfinite(r['fit']) else f"{'-':>7}"
            comb = f"{r['combined']:7.3f}" if np.isfinite(r['combined']) else f"{'inf':>7}"
            print(f"{r['construct']:16s}{r['label']:32s}{fit}{r['penalty']:6.2f}{comb}"
                  f"{r['mcs_hi']:7.2f}  {','.join(r['fired']) if r['fired'] else 'clean'}")

    def show(self, construct, label):
        return self.results[construct].show(label)
