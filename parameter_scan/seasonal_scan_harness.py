"""
seasonal_scan_harness.py — dynamical (seasonal-forcing) scan harness for MS3 Fig 4.

Built ON TOP of baseline_r0_seasonal_* + parscan_utils_extended; the steady-state
gradient_scan_harness is left untouched. Where the SS harness sweeps a constant-F_N
gradient and tail-means each point, this runs ONE long seasonal IVP per
(allometry x forcing x fish x params) combo and reduces it to a post-spin-up
long-run summary, with dynamical flags.

Design (2026-06-22, with Benny):
- Allometry is a switchable CONSTRUCT axis: growth = Taniguchi / Banas / Maranon+Ward
  (grazing stays Dutkiewicz, per the functional-response split).
- Parameters (mP, m_Z, grazing KsZ/sigma_log) are a scannable `param_grid` axis, so a
  large multi-parameter test is a declaration, not a rewrite.
- Forcing = per-era seasonal climatologies (F_N, d_e, T) from the obs monthly CSV
  (d_e/T forced DIRECTLY from obs, matching the 2026-06-22 SeasonalForcing change).
- COMPARATOR (Benny, 2026-06-22): compare BOTH model-mean vs obs-mean AND model-median
  vs obs-median; the headline `score` is the MAX of the two distances, so a combo ranks
  well only if it hits the obs on BOTH statistics ("if it hits both, we're good").
- Export is converted to an AREAL flux (rate * d_e(t) = w_sink*D) to match the obs
  sediment-trap units (mmol N m-2 d-1). N / PP / Z compared in their native units.
- Flags ANNOTATE, never filter. A run that NaN-terminates early has INVALID long-run
  stats: its metrics are nulled and `has_nan` (+ `nan_frac`) flag it -- no misleading
  partial-window means.
- Incremental save: every completed run is appended to disk.

Benny's norms: solver settings surfaced (SEASONAL_SOLVER_KWARGS printed, not baked into
a setup); every scan re-runs fresh; large-Z + export magnitudes are SOFT (judge pulse /
direction). Nothing is auto-labelled a good fit -- the table shows every metric.
"""
import time
import pickle
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xso
from xso.parscans import run_parallel_tasks
import parscan_utils_extended as pue
from cariaco_obs import DEFAULT_CSV_PATH
from baseline_r0_seasonal_comps import _build_fn_func
from baseline_r0_seasonal_setups import (
    model_baseline_seasonal, make_seasonal_input_vars,
    SLIM_OUTPUT_VARS, IVP_SOLVER_KWARGS, SPLINE_K,
)
from baseline_r0_setups import phyto_esd, M_P, M_Z_BULK

GEOM = np.asarray(pue.CARIACO_PHYTO_BIN_GEOMEANS, float)        # [0.63, 6.3, 63] um
_MEDGES = np.cumsum([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])  # month day-bounds
PERIOD = 365.0

# Stiff blooming runs trip on tiny transient negative fluxes; loosen the instability
# floor at run-time (handover 2026-06-22). Surfaced (printed), not baked into a setup;
# genuine blow-ups still propagate NaN -> has_nan.
SEASONAL_SOLVER_KWARGS = {**IVP_SOLVER_KWARGS, 'instability_neg_threshold': -1e-3}

# Composition targets entering the numeric score (mcs + the 3 fractions).
SCORE_KEYS = ['mcs', 'pico', 'nano', 'micro']
# All metrics carried per run.
METRIC_KEYS = ['mcs', 'pico', 'nano', 'micro', 'Z200', 'Z500', 'sumP', 'N', 'PP', 'Export', 'D']

# param_grid flat keys -> (iv slot, param). mP / m_Z are handled as make_seasonal_input_vars
# args; everything below is applied as a post-build iv override.
_PARAM_SLOT = {
    'KsZ': ('Grazing', 'KsZ'), 'sigma_log': ('Grazing', 'sigma_log'),
    'GGE': ('GrazingRouter', 'gge'),                 # gross growth efficiency (Stock 0.25 default)
    'k_remin': ('DetritusRemin', 'k_remin'),         # remineralisation rate [d-1]
    'w_sink': ('DetritusSink', 'w_sink'),            # detritus sinking velocity [m d-1]
    'm_Zlin': ('ZooLinMortality', 'rate'),           # linear zoo loss [d-1]
    'graze_fD': ('GrazingRouter', 'frac_D'), 'graze_fX': ('GrazingRouter', 'frac_export'),
    'zq_fD': ('ZooQuadMortality', 'frac_D'), 'zq_fX': ('ZooQuadMortality', 'frac_export'),
    'pm_fD': ('PhytoMortality', 'frac_D'), 'pm_fX': ('PhytoMortality', 'frac_export'),
}

ERA_OF = lambda y: 'pre' if y < 2005 else ('post' if y < 2014 else 'recovery')
GROUP_ERAS = {'pre': ['pre'], 'post': ['post'], 'recovery': ['recovery'],
              'pre+recovery': ['pre', 'recovery'], 'post+recovery': ['post', 'recovery']}


# =============================================================================
# Allometries (GROWTH only; grazing stays Dutkiewicz). Single-source unless noted.
# =============================================================================
def allometry(name, esd=phyto_esd):
    e = np.asarray(esd, float)
    if name == 'taniguchi':                       # Taniguchi 2014 Table 1 (single-source)
        return dict(name=name, mu_max=1.36 * e ** (-0.16), halfsat=0.33 * e ** (0.48))
    if name == 'banas':                           # Banas 2011 (single-source growth)
        # NB native Banas pairs with size-scaled mortality (0.1*mu_max); this harness
        # keeps the uniform-mP model wiring, so a Banas row is a GROWTH-only swap.
        return dict(name=name, mu_max=2.6 * e ** (-0.45), halfsat=0.1 * e ** (1.0))
    if name == 'maranon_ward':                    # Maranon 2013 mu (unimodal) + Ward 2012 K_s
        # NB TWO-SOURCE growth splice (Maranon gives no K_s) -- flagged vs single-source.
        mu = np.where(e <= 5.38, 0.33 * e ** (0.57), 1.83 * e ** (-0.45))
        return dict(name=name, mu_max=mu, halfsat=0.143 * e ** (0.81))
    raise ValueError(f"unknown allometry {name!r}")


DEFAULT_CONSTRUCTS = ['taniguchi', 'maranon_ward', 'banas']


# =============================================================================
# Obs side — per-era seasonal forcings + the per-era fingerprint (median AND mean)
# =============================================================================
def _monthly_mean(df, eras, col):
    s = (df[df['era'].isin(eras)].dropna(subset=[col])
         .groupby('mo')[col].mean().reindex(range(1, 13)))
    return s.interpolate(limit_direction='both').values        # gaps -> spline-friendly


def build_forcings(groups=('pre+recovery', 'post', 'recovery'), csv=DEFAULT_CSV_PATH):
    """Per-era 12-month F_N / d_e / T climatologies (d_e, T forced directly from obs)."""
    df = pd.read_csv(csv, parse_dates=['date'])
    df['era'] = df['date'].dt.year.map(ERA_OF)
    df['mo'] = df['date'].dt.month
    return {g: dict(fn=_monthly_mean(df, GROUP_ERAS[g], 'FN_mmolN_m2_d'),
                    de=_monthly_mean(df, GROUP_ERAS[g], 'depth_cutoff'),
                    t=_monthly_mean(df, GROUP_ERAS[g], 'Temp_C')) for g in groups}


def _obs_fingerprint(d, stat):
    """One era's obs fingerprint at `stat` ('median' or 'mean'): mcs, Pico/Nano/Micro
    fraction, sumP, N, PP, Export, Z>200, Z>500 -- each in its native obs units."""
    binc = ['pico_mmolN', 'nano_mmolN', 'micro_mmolN']
    agg = (lambda s: float(np.nanmedian(s))) if stat == 'median' else (lambda s: float(np.nanmean(s)))
    fb = d.dropna(subset=binc)
    tot = fb[binc].sum(axis=1).to_numpy()
    fr = fb[binc].to_numpy() / np.maximum(tot[:, None], 1e-30)         # (n_months, 3)
    mcs_month = 10.0 ** (fr @ np.log10(GEOM))
    return dict(
        mcs=agg(pd.Series(mcs_month)),
        pico=agg(fr[:, 0]), nano=agg(fr[:, 1]), micro=agg(fr[:, 2]),
        sumP=agg(tot),
        N=agg(d['NO3_mmolN'].dropna()),
        PP=agg(d['PP_mmolN_m3_d'].dropna()),
        Export=agg(d['export_flux_corrected_mmolN'].dropna()),
        Z200=agg(d['zoo_gt200_mmolN'].dropna()),
        Z500=agg(d['zoo_gt500_mmolN'].dropna()))


def build_obs_targets(groups=('pre+recovery', 'post', 'recovery'), csv=DEFAULT_CSV_PATH,
                      regimes=None, forcing_complete=True):
    """Per-era obs fingerprint at BOTH statistics: {group: {'med': {...}, 'mean': {...}}}.
    `regimes` optionally restricts to a regime_adjusted whitelist (default None = all months).
    `forcing_complete` (default True) restricts to months with F_N/d_e/T all present (the
    forcing's data basis -- the fair model-obs comparison set)."""
    df = pd.read_csv(csv, parse_dates=['date'])
    df['era'] = df['date'].dt.year.map(ERA_OF)
    if regimes is not None and 'regime_adjusted' in df.columns:
        df = df[df['regime_adjusted'].isin(regimes)]
    if forcing_complete:
        _fc = [c for c in ('FN_mmolN_m2_d', 'depth_cutoff', 'Temp_C') if c in df.columns]
        df = df[df[_fc].notna().all(axis=1)]
    out = {}
    for g in groups:
        d = df[df['era'].isin(GROUP_ERAS[g])]
        out[g] = {'med': _obs_fingerprint(d, 'median'), 'mean': _obs_fingerprint(d, 'mean')}
    return out


def build_obs_monthly(groups=('pre+recovery', 'post', 'recovery'), csv=DEFAULT_CSV_PATH,
                      regimes=None, forcing_complete=True):
    """Per-era RAW monthly obs rows -- the points behind the clouds / boxplots that
    build_obs_targets only aggregates. Returns {group: DataFrame} with a 'mo' column,
    each metric in native obs units, and a 'regime' column (regime_adjusted) so plots can
    mark transition / unclassified months. `regimes` (e.g. ['upwelling','relaxed']) keeps
    only those rows. `forcing_complete` (default True) keeps only months with F_N, d_e
    (depth_cutoff) and T (Temp_C) all present -- the months that defined the model forcing,
    i.e. the fair model-obs comparison set."""
    df = pd.read_csv(csv, parse_dates=['date'])
    df['era'] = df['date'].dt.year.map(ERA_OF)
    df['mo'] = df['date'].dt.month
    has_reg = 'regime_adjusted' in df.columns
    if regimes is not None and has_reg:
        df = df[df['regime_adjusted'].isin(regimes)]
    if forcing_complete:
        _fc = [c for c in ('FN_mmolN_m2_d', 'depth_cutoff', 'Temp_C') if c in df.columns]
        df = df[df[_fc].notna().all(axis=1)]
    binc = ['pico_mmolN', 'nano_mmolN', 'micro_mmolN']
    out = {}
    for g in groups:
        d = df[df['era'].isin(GROUP_ERAS[g])]
        A = d[binc].to_numpy(float)
        tot = A.sum(1)
        with np.errstate(invalid='ignore', divide='ignore'):
            fr = np.where(tot[:, None] > 0, A / tot[:, None], np.nan)
        out[g] = pd.DataFrame({
            'mo': d['mo'].to_numpy(),
            'mcs': np.where(tot > 0, 10.0 ** (fr @ np.log10(GEOM)), np.nan),
            'pico': fr[:, 0], 'nano': fr[:, 1], 'micro': fr[:, 2],
            'sumP': np.where(tot > 0, tot, np.nan),
            'N': d['NO3_mmolN'].to_numpy(),
            'PP': d['PP_mmolN_m3_d'].to_numpy(),
            'Export': d['export_flux_corrected_mmolN'].to_numpy(),
            'Z200': d['zoo_gt200_mmolN'].to_numpy(),
            'Z500': d['zoo_gt500_mmolN'].to_numpy(),
            'FN': d['FN_mmolN_m2_d'].to_numpy(),
            'regime': (d['regime_adjusted'].to_numpy() if has_reg else np.full(len(d), np.nan, object)),
        })
    return out


# =============================================================================
# Run + reduce one seasonal IVP -> dynamical long-run summary
# =============================================================================
def _reduce(out):
    """Raw IVP output -> metric time series. Export is the per-volume sinking RATE here;
    run_one converts it to an areal flux via d_e(t)."""
    pb, zb = out['Phytoplankton__biomass'], out['Zooplankton__biomass']
    pdim = [d for d in pb.dims if d != 'time'][0]
    zdim = [d for d in zb.dims if d != 'time'][0]
    pe, ze = out[pdim].values, out[zdim].values
    P = pb.transpose(pdim, 'time').values
    Z = zb.transpose(zdim, 'time').values
    W = pue.compute_sieburth_weights(pe)
    bins = W @ P
    sb = bins.sum(0)
    fr = np.divide(bins, sb, out=np.full_like(bins, np.nan), where=sb > 0)
    res = dict(t=out['time'].values, sumP=P.sum(0), mcs=10 ** (np.log10(GEOM) @ fr),
               pico=fr[0], nano=fr[1], micro=fr[2],
               Z200=Z[ze > 200].sum(0), Z500=Z[ze > 500].sum(0),
               N=out['Nutrient__value'].values)
    res['minP'] = P.min(0)    # per-timestep min over classes -> floor-margin (robustness)
    res['minZ'] = Z.min(0)
    res['Ztot'] = Z.sum(0)    # total zooplankton (NPZD state time series)
    for src, key in [('Growth__uptake_value', 'PP'),
                     ('DetritusSink__sinking_value', 'Export'),  # per-volume rate (-> areal in run_one)
                     ('Detritus__value', 'D')]:
        if src in out:
            da = out[src]
            extra = [d for d in da.dims if d != 'time']
            res[key] = (da.sum(extra) if extra else da).values
    return res


def _clim(r, spinup_yr):
    """Post-spin-up long-run MEAN + MEDIAN + interannual SD + dynamical flags. If the run
    NaN-terminated, long-run stats are invalid -> nulled, flagged by has_nan/nan_frac."""
    keep = r['t'] >= spinup_yr * 365.0
    sp = r['sumP'][keep]
    has_nan = bool(np.isnan(sp).any() or np.isnan(r['Z200'][keep]).any())
    keys = [k for k in METRIC_KEYS if k in r]
    s = {'has_nan': has_nan}
    if has_nan:
        s['nan_frac'] = float(np.isnan(sp).mean())
        for k in keys:
            s[k] = s[k + '_med'] = s[k + '_sd'] = np.nan
            s['clim_' + k] = np.full(12, np.nan)
        s['Z200_peak'] = s['cv_sumP'] = s['mcs_conv'] = s['minP'] = s['minZ'] = np.nan
        return s
    mo = np.clip(np.searchsorted(_MEDGES, np.mod(r['t'][keep], 365.0), 'right') + 1, 1, 12)
    yr = (r['t'][keep] // 365.0).astype(int)
    ys = np.unique(yr)
    half = len(ys) // 2
    for k in keys:
        v = r[k][keep]
        s['clim_' + k] = np.array([np.nanmean(v[mo == m]) if np.any(mo == m) else np.nan
                                   for m in range(1, 13)])
        s[k] = float(np.nanmean(v))                 # long-run MEAN  (all post-spin-up days)
        s[k + '_med'] = float(np.nanmedian(v))      # long-run MEDIAN
        ann = np.array([np.nanmean(v[yr == y]) for y in ys])
        s[k + '_sd'] = float(np.nanstd(ann))        # interannual SD (std of annual means)
    s['Z200_peak'] = float(np.nanmax(s['clim_Z200']))
    s['cv_sumP'] = float(np.nanstd(sp) / max(np.nanmean(sp), 1e-12))
    s['minP'] = float(np.nanmin(r['minP'][keep])) if 'minP' in r else np.nan
    s['minZ'] = float(np.nanmin(r['minZ'][keep])) if 'minZ' in r else np.nan
    am = np.array([np.nanmean(r['mcs'][keep][yr == y]) for y in ys])
    s['mcs_conv'] = float(np.nanmean(am[half:]) / max(np.nanmean(am[:half]), 1e-12))
    return s


def run_one(construct, forcing, fish_rate=0.0, years=60, spinup=15, spline_s=0.0,
            mP=None, m_Z=None, grazing=None, iv_overrides=None,
            solver_kwargs=SEASONAL_SOLVER_KWARGS, return_traj=False):
    """One seasonal IVP. Growth = construct; mP / m_Z / grazing override the defaults;
    iv_overrides = {slot: {param: val}} sets arbitrary closure/remin/routing params.
    return_traj=True returns (clim, r) keeping the full post-reduce trajectory r
    (metric time series + forcing FN(t)/de(t)/T(t) attached) for diagnostics; default
    returns the clim summary only, so the scan path is byte-identical."""
    iv = make_seasonal_input_vars(
        forcing['fn'], forcing['de'], forcing['t'], fish_rate=fish_rate,
        mu_max=construct['mu_max'], halfsat=construct['halfsat'],
        mP=(M_P if mP is None else mP), m_Z=(M_Z_BULK if m_Z is None else m_Z),
        spline_s=spline_s)
    if grazing:
        iv['Grazing'].update(grazing)
    for slot, d in (iv_overrides or {}).items():
        iv[slot].update(d)
    time_ax = np.arange(0.0, years * 365.0 + 1.0, 1.0)
    setup = xso.setup(solver='solve_ivp', model=model_baseline_seasonal, time=time_ax,
                      input_vars=iv, output_vars=SLIM_OUTPUT_VARS, solver_kwargs=solver_kwargs)
    out = pue.run_single_point(model_baseline_seasonal, setup, {})
    r = _reduce(out)
    if 'Export' in r:                                # per-volume rate -> areal flux (= w_sink*D)
        de_t = _build_fn_func(forcing['de'], PERIOD, SPLINE_K, spline_s)(r['t'])
        r['Export'] = r['Export'] * de_t
    clim = _clim(r, spinup)
    if return_traj:                                  # attach forcing series for diagnostics
        r['FN'] = _build_fn_func(forcing['fn'], PERIOD, SPLINE_K, spline_s)(r['t'])
        r['de'] = _build_fn_func(forcing['de'], PERIOD, SPLINE_K, spline_s)(r['t'])
        r['T'] = _build_fn_func(forcing['t'], PERIOD, SPLINE_K, spline_s)(r['t'])
        return clim, r
    return clim


# =============================================================================
# Scoring — itemised composition distance; one per statistic
# =============================================================================
def score(model_vals, obs_vals, weights=None):
    """Normalised-RMS distance over SCORE_KEYS (mcs + fractions)."""
    w = weights or {k: 1.0 for k in SCORE_KEYS}
    terms = {k: (model_vals[k] - obs_vals[k]) / obs_vals[k]
             for k in w
             if obs_vals.get(k) and np.isfinite(model_vals.get(k, np.nan)) and abs(obs_vals[k]) > 1e-9}
    if not terms:
        return np.nan
    return float(np.sqrt(sum(w[k] * terms[k] ** 2 for k in terms) / sum(w[k] for k in terms)))


# =============================================================================
# Scan driver (incremental save) + result wrapper
# =============================================================================
def _seasonal_worker(kw):
    """One combo -> run_one's _clim summary dict (carries has_nan for NaN-terminated
    runs). A genuine exception propagates; run_parallel_tasks' worker wrapper catches
    it and returns an error sentinel, so the pool keeps going."""
    return run_one(**kw)


def run_seasonal_scan(constructs=DEFAULT_CONSTRUCTS,
                      groups=('pre+recovery', 'post', 'recovery'),
                      fish_rates=(0.0,), param_grid=None, years=60, spinup=15,
                      spline_s=0.0, solver_kwargs=SEASONAL_SOLVER_KWARGS,
                      save_path='seasonal_scan_results.pkl', progress=True,
                      processes=None):
    """Cartesian product (construct x group x fish x param_grid) of seasonal IVPs,
    run in parallel across `processes` workers (default os.cpu_count()-1).
    `param_grid` = {param: [values]} where param in {'mP','m_Z','KsZ','sigma_log'}.
    score = max(model-mean-vs-obs-mean, model-median-vs-obs-median). Each completed
    run is appended + re-pickled from the parent (single writer)."""
    forcings = build_forcings(groups)
    obs = build_obs_targets(groups)
    specs = [allometry(c) if isinstance(c, str) else c for c in constructs]
    pnames = list((param_grid or {}).keys())
    pgrid = [dict(zip(pnames, pv)) for pv in itertools.product(*[param_grid[p] for p in pnames])] \
        if pnames else [{}]
    combos = [(s, g, f, p) for s in specs for g in groups for f in fish_rates for p in pgrid]
    n = len(combos)

    # resolve one combo -> run_one kwargs (mP/m_Z direct; the rest -> iv_overrides)
    def _kwargs(s, g, f, p):
        ivo = {}
        for k, v in p.items():
            if k in ('mP', 'm_Z'):
                continue
            slot, key = _PARAM_SLOT[k]
            ivo.setdefault(slot, {})[key] = v
        return dict(construct=s, forcing=forcings[g], fish_rate=f, years=years,
                    spinup=spinup, spline_s=spline_s, mP=p.get('mP'), m_Z=p.get('m_Z'),
                    iv_overrides=(ivo or None), solver_kwargs=solver_kwargs)

    tasks = [(_kwargs(s, g, f, p),) for (s, g, f, p) in combos]   # 1-tuple per task

    t0 = time.time()
    print(f"[seasonal scan] {n} runs: {[s['name'] for s in specs]} x {list(groups)} x "
          f"fish={list(fish_rates)} x params{pnames or '[]'} | {years} yr each (spin-up {spinup})")
    print(f"solver: {solver_kwargs} | spline_s={spline_s} | processes={processes or 'cpu-1'}")

    records = []

    # parent-side, single writer: score + assemble + incremental save + rich line
    def on_result(item, done, n):
        s, g, f, p = combos[item['index']]
        if item['ok']:
            m = item['result']
            sc_med = score({k: m.get(k + '_med', np.nan) for k in SCORE_KEYS}, obs[g]['med'])
            sc_mean = score({k: m.get(k, np.nan) for k in SCORE_KEYS}, obs[g]['mean'])
            sc = float(np.nanmax([sc_med, sc_mean])) if np.isfinite([sc_med, sc_mean]).all() else np.nan
            rec = dict(construct=s['name'], group=g, fish=float(f), **p,
                       score=sc, score_med=sc_med, score_mean=sc_mean, **m)
            for k in obs[g]['med']:
                rec[f'obs_{k}_med'] = obs[g]['med'][k]
                rec[f'obs_{k}_mean'] = obs[g]['mean'][k]
        else:
            rec = dict(construct=s['name'], group=g, fish=float(f), **p,
                       score=np.inf, has_nan=True, error=item['error'])
        records.append(rec)
        if save_path:
            with open(save_path, 'wb') as fh:
                pickle.dump(dict(records=records, obs=obs, forcings=forcings,
                                 param_names=pnames), fh)
        if progress:
            el = time.time() - t0
            print(f"  {done:3d}/{n} {s['name']:13s} {g:13s} fish={f:<4g} {p} "
                  f"mcs={rec.get('mcs', np.nan):5.2f}/{rec.get('mcs_med', np.nan):5.2f} "
                  f"micro={rec.get('micro', np.nan):4.2f} Z200pk={rec.get('Z200_peak', np.nan):5.3f} "
                  f"conv={rec.get('mcs_conv', np.nan):4.2f} nan={rec.get('has_nan')} "
                  f"score={rec.get('score', np.nan):5.2f} | {el/60:4.1f}m "
                  f"ETA {el/done*(n-done)/60:4.1f}m", flush=True)

    # driver prints errors + the end summary (with has_nan tally); our rich line
    # above is the per-combo detail, so the driver's generic line is off.
    run_parallel_tasks(_seasonal_worker, tasks, processes=processes,
                       on_result=on_result, progress=False,
                       tally_flags=['has_nan'], label='seasonal')

    return SeasonalScanResult(records, obs, forcings, pnames)


def load_results(save_path='seasonal_scan_results.pkl'):
    with open(save_path, 'rb') as fh:
        d = pickle.load(fh)
    return SeasonalScanResult(d['records'], d['obs'], d['forcings'], d.get('param_names', []))


class SeasonalScanResult:
    """Records + obs (median & mean) + forcings. Tables show model vs obs on BOTH stats."""
    def __init__(self, records, obs, forcings, param_names=()):
        self.records = records
        self.obs = obs
        self.forcings = forcings
        self.param_names = list(param_names)
        self.df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith('clim_')}
                                for r in records])

    def table(self, sort='score'):
        """Headline: scores (mean/median/combined) + mcs & Micro on both stats + flags."""
        cols = (['construct', 'group', 'fish'] + self.param_names +
                ['score', 'score_mean', 'score_med',
                 'mcs', 'mcs_med', 'micro', 'micro_med',
                 'N', 'Z200_peak', 'cv_sumP', 'mcs_conv', 'has_nan'])
        d = self.df[[c for c in cols if c in self.df.columns]].copy()
        return d.sort_values(sort) if sort in d.columns else d

    def compare(self, metric='mcs', sort='score'):
        """Model (mean & median) vs obs (mean & median) for one metric, across combos."""
        cols = (['construct', 'group', 'fish'] + self.param_names +
                [metric, metric + '_med',
                 f'obs_{metric}_mean', f'obs_{metric}_med', 'has_nan', 'score'])
        d = self.df[[c for c in cols if c in self.df.columns]].copy()
        return d.sort_values(sort) if sort in d.columns else d

    def fish_effect(self):
        """Per (construct, group, params): fish-on (strongest r_F) minus fish-off."""
        key = ['construct', 'group'] + self.param_names
        rows = []
        for vals, sub in self.df.groupby(key):
            off, on = sub[sub.fish == 0], sub[sub.fish > 0]
            if len(off) and len(on):
                o, n = off.iloc[0], on.sort_values('fish').iloc[-1]
                row = dict(zip(key, vals if isinstance(vals, tuple) else (vals,)))
                row.update(fish_on=n.fish, d_mcs=n.mcs - o.mcs, d_micro=n.micro - o.micro,
                           Z200_off=o.Z200, Z200_on=n.Z200, Z200_drop=o.Z200 - n.Z200)
                rows.append(row)
        return pd.DataFrame(rows)

    def _rec(self, **sel):
        for r in self.records:
            if all(abs(r.get(k) - v) < 1e-9 if isinstance(v, (int, float)) else r.get(k) == v
                   for k, v in sel.items()):
                return r
        return None

    def plot_clim(self, keys=('Z200', 'mcs', 'micro'), fishes=(0.0,), **sel):
        """Seasonal climatology for a selected combo (e.g. construct=, group=, mP=), one
        line per fish rate."""
        labs = {'Z200': 'Zoo>200', 'mcs': 'mean cell size (um)', 'micro': 'Micro fraction',
                'sumP': 'sumP', 'N': 'N', 'Export': 'Export (areal)'}
        fig, axes = plt.subplots(1, len(keys), figsize=(3.7 * len(keys), 3.3))
        axes = np.atleast_1d(axes)
        for ax, k in zip(axes, keys):
            for f in fishes:
                rec = self._rec(fish=float(f), **sel)
                if rec and 'clim_' + k in rec:
                    ax.plot(range(1, 13), rec['clim_' + k], '-o', ms=3, label=f"fish={f:g}")
            ax.set_xlabel('month'); ax.set_title(labs.get(k, k), fontsize=9)
            ax.set_xticks(range(1, 13))
        axes[0].legend(fontsize=7, frameon=False)
        fig.suptitle(' | '.join(f"{k}={v}" for k, v in sel.items()), fontsize=9)
        fig.tight_layout()
        return fig
