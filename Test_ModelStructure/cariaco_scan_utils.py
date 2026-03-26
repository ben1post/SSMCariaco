"""
cariaco_scan_utils.py
=====================
Reusable utilities for CARIACO NPxZxf model–data comparison.

Provides:
  - Observational targets and parameter definitions
  - Model output → observational-bin aggregation
  - Cost function (log-ratio)
  - Parallel model evaluation (1-D scans and random sampling)
  - Plotting helpers
"""

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from IPython.display import clear_output

from npxzxf_models import phyto_esd, zoo_esd, n_classes
from allometric_params import compute_grazing_kernel, compute_gge

plt.rcParams.update({
    'figure.dpi': 120,
    'axes.grid': True,
    'grid.alpha': 0.3,
})


# ═════════════════════════════════════════════════════════════════════
#  OBSERVATIONAL TARGETS  (mmol N m⁻³)
# ═════════════════════════════════════════════════════════════════════

OBS_TARGETS = {
    'micro_phyto': {'mean': 0.15627, 'median': 0.09544},
    'nano_phyto':  {'mean': 0.12162, 'median': 0.11883},
    'pico_phyto':  {'mean': 0.17506, 'median': 0.17513},
    'zoo_gt200':   {'mean': 0.05934, 'median': 0.05304},
    'zoo_gt500':   {'mean': 0.03280, 'median': 0.02617},
    'NO3':         {'mean': 2.01581, 'median': 1.63575},
}


# ═════════════════════════════════════════════════════════════════════
#  PARAMETER REGISTRY
# ═════════════════════════════════════════════════════════════════════
#
#  Each entry maps a short, human-readable name to its metadata.
#  - xso_name :  XSO parameter path (None for derived parameters)
#  - default  :  baseline value baked into model_setup_ivp_cariaco
#  - range    :  (lo, hi) sensible bounds for scanning / sampling
#  - label    :  axis label for plots

PARAM_DEFS = {
    # ── direct (scalar → model parameter) ────────────────────────
    'KsZ': {
        'xso_name': 'Grazing__KsZ',
        'default':  3.0,
        'range':    (0.1, 10.0),
        'label':    r'Grazing half-sat $K_Z$ (mmol N m$^{-3}$)',
    },
    'mZ': {
        'xso_name': 'ZooMortality__rate',
        'default':  0.01,
        'range':    (0.001, 0.5),
        'label':    r'Zoo mortality $m_Z$ (d$^{-1}$)',
    },
    'N0': {
        'xso_name': 'N0__value',
        'default':  5.5564,
        'range':    (1.0, 15.0),
        'label':    r'Deep nutrient $N_0$ (mmol N m$^{-3}$)',
    },
    'dilution': {
        'xso_name': 'Inflow__rate',
        'default':  0.016786,
        'range':    (0.005, 0.1),
        'label':    r'Dilution rate $d$ (d$^{-1}$)',
    },
    'fish_rate': {
        'xso_name': 'FishGrazing__rate',
        'default':  0.1,
        'range':    (0.0001, 1.0),
        'label':    r'Fish grazing rate (d$^{-1}$)',
    },

    # ── derived (scalar → recompute array → model parameter) ─────
    'sigma_log': {
        'xso_name': None,
        'default':  0.15,
        'range':    (0.05, 0.5),
        'label':    r'Grazing kernel width $\sigma_{\log}$',
    },
    'gge_small': {
        'xso_name': None,
        'default':  0.35,
        'range':    (0.1, 0.5),
        'label':    r'GGE (small zoo)',
    },
    'gge_large': {
        'xso_name': None,
        'default':  0.15,
        'range':    (0.05, 0.3),
        'label':    r'GGE (large zoo)',
    },
}


# ═════════════════════════════════════════════════════════════════════
#  PARAMETER CONVERSION
# ═════════════════════════════════════════════════════════════════════

def params_to_overrides(param_dict):
    """Convert a {friendly_name: value} dict into XSO model overrides.

    Handles both direct parameters (scalar → XSO name) and derived
    parameters that require recomputing arrays (sigma_log, gge_*).

    Parameters
    ----------
    param_dict : dict
        Keys are names from PARAM_DEFS, values are scalars.

    Returns
    -------
    dict
        Keys are XSO parameter paths, values are scalars or arrays
        ready to pass to model_setup.xsimlab.update_vars().
    """
    overrides = {}

    # 1. Direct parameters
    for name, value in param_dict.items():
        pdef = PARAM_DEFS.get(name)
        if pdef and pdef['xso_name'] is not None:
            overrides[pdef['xso_name']] = value

    # 2. Grazing kernel (depends on sigma_log)
    if 'sigma_log' in param_dict:
        overrides['Grazing__phiPZ'] = compute_grazing_kernel(
            phyto_esd, zoo_esd,
            theta_opt=10.0,
            sigma_log=param_dict['sigma_log'],
        )

    # 3. GGE array (depends on gge_small and/or gge_large)
    if 'gge_small' in param_dict or 'gge_large' in param_dict:
        gge_s = param_dict.get('gge_small', PARAM_DEFS['gge_small']['default'])
        gge_l = param_dict.get('gge_large', PARAM_DEFS['gge_large']['default'])
        overrides['GGE__gge'] = compute_gge(zoo_esd, gge_small=gge_s, gge_large=gge_l)

    return overrides


# ═════════════════════════════════════════════════════════════════════
#  AGGREGATION  (model spectrum → observational bins)
# ═════════════════════════════════════════════════════════════════════

def _log_bin_edges(centers):
    q = centers[1] / centers[0]
    h = np.sqrt(q)
    edges = np.zeros(len(centers) + 1)
    edges[0] = centers[0] / h
    edges[1:] = centers * h
    return edges


def _frac_overlap(lo, hi, tmin, tmax):
    omin, omax = max(lo, tmin), min(hi, tmax)
    if omin >= omax:
        return 0.0
    return (np.log10(omax) - np.log10(omin)) / (np.log10(hi) - np.log10(lo))


def aggregate_model(ss_phyto, ss_zoo, ss_nut):
    """Aggregate model size spectra into observational bins.

    Uses module-level phyto_esd / zoo_esd imported from npxzxf_models.

    Returns dict with same keys as OBS_TARGETS.
    """
    p_edges = _log_bin_edges(phyto_esd)
    z_edges = _log_bin_edges(zoo_esd)

    micro = nano = pico = 0.0
    for i in range(len(phyto_esd)):
        b = ss_phyto[i]
        lo, hi = p_edges[i], p_edges[i + 1]
        pico  += b * _frac_overlap(lo, hi, 1e-9, 2.0)
        nano  += b * _frac_overlap(lo, hi, 2.0, 20.0)
        micro += b * _frac_overlap(lo, hi, 20.0, 1e9)

    z200 = z500 = 0.0
    for i in range(len(zoo_esd)):
        b = ss_zoo[i]
        lo, hi = z_edges[i], z_edges[i + 1]
        z200 += b * _frac_overlap(lo, hi, 200.0, 1e9)
        z500 += b * _frac_overlap(lo, hi, 500.0, 1e9)

    return {
        'micro_phyto': micro,
        'nano_phyto':  nano,
        'pico_phyto':  pico,
        'zoo_gt200':   z200,
        'zoo_gt500':   z500,
        'NO3':         float(ss_nut),
    }


# ═════════════════════════════════════════════════════════════════════
#  COST FUNCTION
# ═════════════════════════════════════════════════════════════════════

def compute_cost(model_dict, metric='mean'):
    """Sum of squared log₁₀-ratios between model and observations."""
    cost = 0.0
    for key, obs in OBS_TARGETS.items():
        o = max(obs[metric], 1e-12)
        m = max(model_dict[key], 1e-12)
        cost += (np.log10(m) - np.log10(o)) ** 2
    return cost


# ═════════════════════════════════════════════════════════════════════
#  PARALLEL EXECUTION
# ═════════════════════════════════════════════════════════════════════

_worker_model = None
_worker_setup = None


def _init_worker():
    """Import model objects once per worker process."""
    global _worker_model, _worker_setup
    from npxzxf_models import model, model_setup_ivp_cariaco
    _worker_model = model
    _worker_setup = model_setup_ivp_cariaco


def _eval_point(param_dict):
    """Run one model evaluation; called inside a worker process."""
    try:
        overrides = params_to_overrides(param_dict)
        with _worker_model:
            out = _worker_setup.xsimlab.update_vars(
                input_vars=overrides
            ).xsimlab.run()
        out['time'] = out.time.round(9)

        ss_p = out.Phytoplankton__biomass.isel(time=-1).values
        ss_z = out.Zooplankton__biomass.isel(time=-1).values
        ss_n = float(out.Nutrient__value.isel(time=-1).values)

        agg  = aggregate_model(ss_p, ss_z, ss_n)
        cost = compute_cost(agg)

        return dict(params=param_dict, cost=cost, agg=agg,
                    ss_phyto=ss_p, ss_zoo=ss_z, ss_nut=ss_n)

    except Exception as e:
        return dict(params=param_dict, cost=np.inf,
                    agg={k: np.nan for k in OBS_TARGETS},
                    ss_phyto=np.full(n_classes, np.nan),
                    ss_zoo=np.full(n_classes, np.nan),
                    ss_nut=np.nan, error=str(e))


def run_parameter_samples(sample_dicts, n_procs=20, label='Scan'):
    """Evaluate a list of parameter dicts in parallel.

    Parameters
    ----------
    sample_dicts : list[dict]
        Each dict maps friendly parameter names to values.
    n_procs : int
        Number of worker processes.
    label : str
        Label shown in progress output.

    Returns
    -------
    list[dict]
        One result dict per sample (see _eval_point for keys).
    """
    n = len(sample_dicts)
    results = []

    with Pool(processes=n_procs, initializer=_init_worker) as p:
        for i, result in enumerate(p.imap_unordered(_eval_point, sample_dicts), 1):
            results.append(result)
            clear_output(wait=True)
            print(f"{label}: {i}/{n} complete")

    return results


def evaluate_single(param_dict):
    """Run a single model point without multiprocessing (for quick tests)."""
    from npxzxf_models import model, model_setup_ivp_cariaco

    overrides = params_to_overrides(param_dict)
    with model:
        out = model_setup_ivp_cariaco.xsimlab.update_vars(
            input_vars=overrides
        ).xsimlab.run()
    out['time'] = out.time.round(9)

    ss_p = out.Phytoplankton__biomass.isel(time=-1).values
    ss_z = out.Zooplankton__biomass.isel(time=-1).values
    ss_n = float(out.Nutrient__value.isel(time=-1).values)

    agg  = aggregate_model(ss_p, ss_z, ss_n)
    cost = compute_cost(agg)

    return dict(params=param_dict, cost=cost, agg=agg,
                ss_phyto=ss_p, ss_zoo=ss_z, ss_nut=ss_n)


# ═════════════════════════════════════════════════════════════════════
#  CONVENIENCE:  1-D SCAN
# ═════════════════════════════════════════════════════════════════════

def run_1d_scan(param_name, values, fixed_params=None, n_procs=20):
    """Sweep one parameter over *values*, keeping others at defaults.

    Parameters
    ----------
    param_name : str
        Key in PARAM_DEFS (e.g. 'KsZ', 'sigma_log').
    values : array-like
        Values to scan.
    fixed_params : dict, optional
        Additional parameter overrides held constant across the scan.
    n_procs : int
        Number of workers.

    Returns
    -------
    list[dict]
        Sorted by the scanned parameter value.
    """
    base = fixed_params.copy() if fixed_params else {}
    samples = [{**base, param_name: v} for v in values]

    results = run_parameter_samples(
        samples, n_procs=n_procs,
        label=f'1-D scan ({param_name})',
    )
    results.sort(key=lambda r: r['params'][param_name])
    return results


# ═════════════════════════════════════════════════════════════════════
#  CONVENIENCE:  RANDOM SAMPLING
# ═════════════════════════════════════════════════════════════════════

def generate_random_samples(search_space, n=100, seed=None,
                            log_params=None, method='lhs'):
    """Generate parameter samples for Monte-Carlo exploration.

    Parameters
    ----------
    search_space : dict
        {param_name: (lo, hi)} for each parameter to sample.
    n : int
        Number of samples.
    seed : int, optional
        Random seed for reproducibility.
    log_params : list[str], optional
        Parameters to sample in log₁₀-space (e.g. 'mZ').
    method : str
        'random' for uniform, 'lhs' for Latin-Hypercube (requires scipy).

    Returns
    -------
    list[dict]
        Each dict maps param names to sampled values.
    """
    log_params = log_params or []
    names = list(search_space.keys())
    bounds = [search_space[n] for n in names]
    d = len(names)

    # generate unit-cube samples
    if method == 'lhs':
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=d, seed=seed)
        unit = sampler.random(n)
    else:
        rng = np.random.default_rng(seed)
        unit = rng.random((n, d))

    # scale to parameter bounds
    samples = []
    for row in unit:
        s = {}
        for j, name in enumerate(names):
            lo, hi = bounds[j]
            if name in log_params:
                s[name] = 10 ** (np.log10(lo) + row[j] * (np.log10(hi) - np.log10(lo)))
            else:
                s[name] = lo + row[j] * (hi - lo)
        samples.append(s)
    return samples


# ═════════════════════════════════════════════════════════════════════
#  RESULT HELPERS
# ═════════════════════════════════════════════════════════════════════

def _extract_1d(results, param_name):
    """Pull arrays out of a sorted 1-D result list."""
    vals  = np.array([r['params'][param_name] for r in results])
    costs = np.array([r['cost'] for r in results])
    agg   = {k: np.array([r['agg'][k] for r in results]) for k in OBS_TARGETS}
    return vals, agg, costs


def print_fit_summary(result):
    """Print a single result's comparison table."""
    r = result
    print(f"Cost = {r['cost']:.4f}\n")
    print(f"  {'Component':>15s}  {'Model':>10s}  {'Obs Mean':>10s}  {'Ratio':>8s}")
    print("  " + "-" * 50)
    for comp in OBS_TARGETS:
        mod = r['agg'][comp]
        obs = OBS_TARGETS[comp]['mean']
        ratio = mod / obs if obs > 0 else np.inf
        print(f"  {comp:>15s}  {mod:10.5f}  {obs:10.5f}  {ratio:8.2f}x")
    print(f"\n  Parameters:")
    for k, v in r['params'].items():
        print(f"    {k:>15s} = {v:.6f}")


def print_top_fits(results, n=10):
    """Rank and print the best *n* results."""
    top = sorted(results, key=lambda r: r['cost'])[:n]
    for rank, r in enumerate(top, 1):
        print(f"\n{'='*60}")
        print(f"  Rank {rank}  |  Cost = {r['cost']:.4f}")
        print(f"{'='*60}")
        for k, v in r['params'].items():
            print(f"    {k:>15s} = {v:.6f}")
        print()
        for comp in OBS_TARGETS:
            mod = r['agg'][comp]
            obs = OBS_TARGETS[comp]['mean']
            ratio = mod / obs if obs > 0 else np.inf
            print(f"    {comp:>15s}:  {mod:.5f}  ({ratio:.2f}x obs)")


# ═════════════════════════════════════════════════════════════════════
#  PLOTTING
# ═════════════════════════════════════════════════════════════════════

_COMP_ORDER  = ['pico_phyto', 'nano_phyto', 'micro_phyto',
                'zoo_gt200',  'zoo_gt500',  'NO3']
_COMP_LABELS = ['Pico (<2 µm)', 'Nano (2–20 µm)', 'Micro (>20 µm)',
                'Zoo >200 µm',  'Zoo >500 µm',    'NO₃']


def plot_1d_components(results, param_name):
    """2×3 panel: each component vs. scanned parameter, with obs lines."""
    vals, agg, costs = _extract_1d(results, param_name)
    label = PARAM_DEFS.get(param_name, {}).get('label', param_name)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for ax, key, clabel in zip(axes.ravel(), _COMP_ORDER, _COMP_LABELS):
        ax.plot(vals, agg[key], 'k-', lw=1.5, label='Model')
        ax.axhline(OBS_TARGETS[key]['mean'],   color='C0', ls='--', lw=1, label='Obs mean')
        ax.axhline(OBS_TARGETS[key]['median'], color='C3', ls=':',  lw=1, label='Obs median')
        ax.set_title(clabel, fontsize=11)
        ax.set_ylabel('mmol N m⁻³')
        if ax is axes.ravel()[0]:
            ax.legend(fontsize=8)
    for ax in axes[1, :]:
        ax.set_xlabel(label)
    fig.suptitle(f'1-D Scan: {param_name}', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()


def plot_cost_curve(results, param_name):
    """Cost vs. parameter value with best-fit marker."""
    vals, _, costs = _extract_1d(results, param_name)
    label = PARAM_DEFS.get(param_name, {}).get('label', param_name)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(vals, costs, 'k-', lw=1.5)
    best_idx = np.argmin(costs)
    ax.axvline(vals[best_idx], color='red', ls='--', alpha=0.7,
               label=f'best = {vals[best_idx]:.4f}')
    ax.set_xlabel(label)
    ax.set_ylabel('Cost  (Σ log-ratio²)')
    ax.set_title(f'Cost landscape — {param_name}')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_spectrum(result, title=None):
    """3-panel bar chart: P spectrum, Z spectrum, NO₃ for one result."""
    r = result
    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(14, 5),
        gridspec_kw={'width_ratios': [3, 3, 1]},
    )

    # ── Phytoplankton ──
    bars = ax1.bar(range(len(phyto_esd)), r['ss_phyto'],
                   edgecolor='k', alpha=0.8)
    for i, e in enumerate(phyto_esd):
        bars[i].set_facecolor(
            'royalblue' if e < 2 else ('gold' if e < 20 else 'firebrick')
        )
    ax1.set_xticks(range(len(phyto_esd)))
    ax1.set_xticklabels([f'{e:.1f}' for e in phyto_esd], rotation=45)
    ax1.set_xlabel('ESD (µm)')
    ax1.set_ylabel('Biomass (mmol N m⁻³)')
    ax1.set_title('Phytoplankton')

    # ── Zooplankton ──
    ax2.bar(range(len(zoo_esd)), r['ss_zoo'],
            color='darkorange', edgecolor='k', alpha=0.8)
    ax2.set_xticks(range(len(zoo_esd)))
    ax2.set_xticklabels([f'{e:.0f}' for e in zoo_esd], rotation=45)
    ax2.set_xlabel('ESD (µm)')
    ax2.set_ylabel('Biomass (mmol N m⁻³)')
    ax2.set_title('Zooplankton')
    # net cutoff guides
    ax2.axvline(np.interp(200, zoo_esd, range(len(zoo_esd))),
                color='gray', ls='--', lw=1, label='200 µm net')
    ax2.axvline(np.interp(500, zoo_esd, range(len(zoo_esd))),
                color='gray', ls=':',  lw=1, label='500 µm net')
    ax2.legend(fontsize=9)

    # ── Nutrient ──
    ax3.bar(['NO₃'], [r['ss_nut']], color='steelblue',
            edgecolor='k', alpha=0.8, width=0.5)
    ax3.axhline(OBS_TARGETS['NO3']['mean'],   color='C0', ls='--', lw=1, label='Obs mean')
    ax3.axhline(OBS_TARGETS['NO3']['median'], color='C3', ls=':',  lw=1, label='Obs median')
    ax3.set_ylabel('mmol N m⁻³')
    ax3.set_title('Nutrient')
    ax3.legend(fontsize=8)

    auto_title = f"cost = {r['cost']:.4f}"
    fig.suptitle(title or auto_title, fontsize=11, y=1.02)
    plt.tight_layout()
    plt.show()
