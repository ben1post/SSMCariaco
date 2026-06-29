"""
baseline_r0_seasonal_setups.py — setups for the seasonal (time-varying forcing)
MS3 Cariaco baseline, for the SS-vs-seasonal dynamics test (2026-06-22).

ADDITIVE to baseline_r0_setups.py — reuses every steady-state component and all
default arrays/scalars unchanged; only the forcing layer differs:

  - 'Forcing'     : SeasonalForcing            (F_N(t) periodic spline; d_e(t), T(t) derived)
  - 'Inflow'      : SeasonalNutrientSupply     (flux F_N(t)/d_e(t) -> N)
  - 'DetritusSink': DetritusSinking_seasonal   (d_e(t) read as a foreign forcing)

vs the SS model's ConstantTemperatureForcing / StockNutrientSupply / DetritusSinking.
Growth, grazing, closures, remineralisation, and fish are imported unchanged.

The SS reference run for the comparison uses the existing constant-forcing model
(model_baseline / setup_baseline_slim) at F_N = the cycle's time-mean.

make_seasonal_input_vars(fn_monthly, ...) builds the input-vars dict for a given
12-month F_N climatology; the run cell supplies the obs climatology and the run
length, then builds the xso.setup (time-axis bound at setup creation).
"""

import numpy as np
import xso
from xso.parscans import avg_tail            # re-export (parity with baseline_r0_setups)
from parscan_utils_extended import avg_tail_stats   # noqa: F401  (re-export)

from baseline_r0_comps import (
    Nutrient, PhytoSizeSpectrum, ZooSizeSpectrum, Detritus,
    MonodGrowth_T, DistributedGrazing_TypeIII_T, DistributedGrazingRouter_route,
    PhytoMortality_route, ZooLinearMortality_route, ZooQuadraticMortality_route,
    DetritusRemineralization, FishGrazing_Kernel_rate, FishGrazing_Kernel_route,
)
from baseline_r0_seasonal_comps import (
    SeasonalForcing, SeasonalNutrientSupply, DetritusSinking_seasonal,
)
from baseline_r0_setups import (
    make_baseline_input_vars, SLIM_OUTPUT_VARS, IVP_SOLVER_KWARGS,
    mu_max_arr, ks_arr, M_P, M_Z_BULK, FISH_RATE, FN_DEFAULT,
)

# Obs F_N -> d_e and F_N -> T linear regressions (clamped). Used by the SS gradient
# scans and for this module's placeholder setup only — the seasonal model now forces
# d_e(t)/T(t) DIRECTLY from the obs monthly climatologies (2026-06-22), not via these.
DE_COEFFS = (55.89, 3.966, 19.0, 69.0)   # (intercept, slope, lo, hi)  d_e = clip(a - b*F_N)
T_COEFFS  = (25.64, 0.414, 22.0, 29.0)   # (intercept, slope, lo, hi)  T   = clip(a - b*F_N)

# Fourier forcing defaults (replaced cubic spline, 2026-06-24).
N_HARMONICS = 2   # annual + semi-annual cycle; standard in physical oceanography
PERIOD      = 365.0

# Legacy spline constants kept for diagnostic comparisons only.
SPLINE_K = 3
SPLINE_S = 0.0

# =============================================================================
# New loss-fate routing — TEST CONFIG (2026-06-28). Used ONLY by the *_routed
# model/setups below; the existing model_baseline_seasonal is untouched, so all
# prior scans/runs are unaffected. Grounds the quadratic closure and sardine
# grazing in the Fasham (1990) / Stock (2008) recycle-vs-export precedents
# instead of the leakier as-built routing:
#   Quadratic closure: 67% N / 0% D / 33% export  (Fasham 1990, JMR 48:591:
#       higher-predator chain, assim 75% + GGE 25%; bypasses the slow D pool).
#   Sardine grazing:   50% N / 0% D / 50% export  (Stock 2008, fmz4=0.5:
#       excretion recycles locally; fish biomass + fast pellets leave the box).
# =============================================================================
ZOO_QUAD_FRAC_D_ROUTED      = 0.0
ZOO_QUAD_FRAC_EXPORT_ROUTED = 0.33
FISH_FRAC_D_ROUTED          = 0.0
FISH_FRAC_EXPORT_ROUTED     = 0.5


# =============================================================================
# Model — seasonal forcing variant of model_baseline
# =============================================================================
model_baseline_seasonal = xso.create({
    'Nutrient':         Nutrient,
    'Phytoplankton':    PhytoSizeSpectrum,
    'Zooplankton':      ZooSizeSpectrum,
    'Detritus':         Detritus,
    'Forcing':          SeasonalForcing,              # publishes fn(t), de(t), T(t)
    'Inflow':           SeasonalNutrientSupply,       # flux F_N(t)/d_e(t) -> N
    'Growth':           MonodGrowth_T,
    'Grazing':          DistributedGrazing_TypeIII_T,
    'GrazingRouter':    DistributedGrazingRouter_route,
    'PhytoMortality':   PhytoMortality_route,
    'ZooLinMortality':  ZooLinearMortality_route,
    'ZooQuadMortality': ZooQuadraticMortality_route,
    'DetritusRemin':    DetritusRemineralization,
    'DetritusSink':     DetritusSinking_seasonal,     # de(t) as foreign forcing
    'FishGrazing':      FishGrazing_Kernel_rate,
})


# =============================================================================
# Model — ROUTED variant (TEST CONFIG, 2026-06-28)
#
# Identical to model_baseline_seasonal EXCEPT the fish slot uses the routable
# FishGrazing_Kernel_route (sardine-grazed N split N/D/export instead of fully
# exported). The quadratic-closure routing change needs NO model-dict change —
# ZooQuadraticMortality_route is already routable, so it is enacted purely as a
# parameter-value override in make_seasonal_input_vars_routed.
# =============================================================================
model_baseline_seasonal_routed = xso.create({
    'Nutrient':         Nutrient,
    'Phytoplankton':    PhytoSizeSpectrum,
    'Zooplankton':      ZooSizeSpectrum,
    'Detritus':         Detritus,
    'Forcing':          SeasonalForcing,
    'Inflow':           SeasonalNutrientSupply,
    'Growth':           MonodGrowth_T,
    'Grazing':          DistributedGrazing_TypeIII_T,
    'GrazingRouter':    DistributedGrazingRouter_route,
    'PhytoMortality':   PhytoMortality_route,
    'ZooLinMortality':  ZooLinearMortality_route,
    'ZooQuadMortality': ZooQuadraticMortality_route,
    'DetritusRemin':    DetritusRemineralization,
    'DetritusSink':     DetritusSinking_seasonal,
    'FishGrazing':      FishGrazing_Kernel_route,     # <- routable fish (vs _rate above)
})


# =============================================================================
# Input-vars builder
# =============================================================================
def make_seasonal_input_vars(fn_monthly, de_monthly, t_monthly, fish_rate=FISH_RATE,
                             mu_max=mu_max_arr, halfsat=ks_arr,
                             mP=M_P, m_Z=M_Z_BULK,
                             period=PERIOD, n_harmonics=N_HARMONICS, doy=None):
    """Input-vars for model_baseline_seasonal.

    Starts from make_baseline_input_vars (all the SS defaults: growth allometry,
    grazing kernel, closures, detritus, fish), then swaps the forcing layer:
    removes the constant Temperature slot, replaces Inflow with the seasonal supply,
    and adds the SeasonalForcing slot carrying the 12-month F_N, d_e and T obs
    climatologies (d_e/T forced DIRECTLY from obs, no longer derived from F_N;
    2026-06-22). Forcing interpolation = truncated Fourier fit (n_harmonics=2,
    annual + semi-annual; replaced cubic spline 2026-06-24). DetritusSink's `de`
    reference is unchanged (label 'de' now resolves to the seasonal d_e(t) forcing).

    Override mu_max / halfsat for a different growth allometry (e.g. Marañón+Ward);
    set grazing K_sZ / sigma by editing iv['Grazing']['KsZ'] / ['sigma_log'] on the
    returned dict.
    """
    iv = make_baseline_input_vars(fish_rate=fish_rate, mu_max=mu_max,
                                  halfsat=halfsat, mP=mP, m_Z=m_Z)
    iv.pop('Temperature')   # constant-T slot removed; SeasonalForcing supplies 'temperature'

    iv['Forcing'] = {
        'month_index': list(range(1, 13)), 'month_label': 'month',
        'fn_monthly': np.asarray(fn_monthly, dtype=float),
        'de_monthly': np.asarray(de_monthly, dtype=float),
        't_monthly':  np.asarray(t_monthly,  dtype=float),
        'fn_label': 'fn', 'de_label': 'de', 'temperature_label': 'temperature',
        'period': float(period), 'n_harmonics': int(n_harmonics),
        'fn_scale': 1.0,   # added 2026-06-25; sweep via param_name='Forcing__fn_scale'
        # per-month mean cruise DOY = Fourier fit positions (added 2026-06-28);
        # None -> 12×NaN -> _build_fourier_func falls back to calendar mid-month (legacy).
        'doy_monthly': (np.full(12, np.nan) if doy is None else np.asarray(doy, dtype=float)),
    }
    iv['Inflow'] = {'var': 'N', 'fn': 'fn', 'de': 'de'}
    return iv


def make_seasonal_input_vars_routed(fn_monthly, de_monthly, t_monthly, fish_rate=FISH_RATE,
                                    mu_max=mu_max_arr, halfsat=ks_arr,
                                    mP=M_P, m_Z=M_Z_BULK,
                                    period=PERIOD, n_harmonics=N_HARMONICS, doy=None):
    """Input-vars for model_baseline_seasonal_routed (TEST CONFIG, 2026-06-28).

    Identical to make_seasonal_input_vars EXCEPT two loss-fate routings are
    overridden to the Fasham/Stock recycle-vs-export precedents:
      - quadratic closure -> 67% N / 0% D / 33% export (Fasham 1990)
      - sardine grazing    -> 50% N / 0% D / 50% export (Stock 2008), plus the
        detritus/nutrient targets the routable FishGrazing_Kernel_route needs.
    Everything else (forcing, growth, grazing, detritus, phyto/linear-zoo
    closures) is inherited unchanged, so any output difference vs the existing
    seasonal baseline is attributable solely to the routing change.
    """
    iv = make_seasonal_input_vars(fn_monthly, de_monthly, t_monthly, fish_rate=fish_rate,
                                  mu_max=mu_max, halfsat=halfsat, mP=mP, m_Z=m_Z,
                                  period=period, n_harmonics=n_harmonics, doy=doy)
    iv['ZooQuadMortality'] = {**iv['ZooQuadMortality'],
                              'frac_D': ZOO_QUAD_FRAC_D_ROUTED,
                              'frac_export': ZOO_QUAD_FRAC_EXPORT_ROUTED}
    iv['FishGrazing'] = {**iv['FishGrazing'],
                         'detritus': 'D', 'nutrient': 'N',
                         'frac_D': FISH_FRAC_D_ROUTED,
                         'frac_export': FISH_FRAC_EXPORT_ROUTED}
    return iv


# =============================================================================
# Run-length helpers + a default (placeholder-climatology) setup
# =============================================================================
SEASONAL_YEARS = 5
seasonal_time = np.arange(0.0, SEASONAL_YEARS * 365.0 + 1.0, 1.0)   # daily, full record (no spin-up discard)

# Placeholder climatology so the module imports and the wiring validates; real runs
# build their own setup with the obs F_N climatology and chosen run length.
_PLACEHOLDER_FN_MONTHLY = np.full(12, FN_DEFAULT)
_de_a, _de_b, _de_lo, _de_hi = DE_COEFFS
_t_a, _t_b, _t_lo, _t_hi = T_COEFFS
_PLACEHOLDER_DE_MONTHLY = np.clip(_de_a - _de_b * _PLACEHOLDER_FN_MONTHLY, _de_lo, _de_hi)
_PLACEHOLDER_T_MONTHLY  = np.clip(_t_a  - _t_b  * _PLACEHOLDER_FN_MONTHLY, _t_lo,  _t_hi)

setup_baseline_seasonal_slim = xso.setup(
    solver='solve_ivp', model=model_baseline_seasonal, time=seasonal_time,
    input_vars=make_seasonal_input_vars(_PLACEHOLDER_FN_MONTHLY,
                                        _PLACEHOLDER_DE_MONTHLY, _PLACEHOLDER_T_MONTHLY,
                                        n_harmonics=N_HARMONICS),
    output_vars=SLIM_OUTPUT_VARS, solver_kwargs=IVP_SOLVER_KWARGS)

# Tight-tolerance slim variant for seasonal 2D parameter scans (added 2026-06-25).
# Mirrors setup_baseline_slim_tight (SS variant). Tight RK45 tols suppress the
# Marañón+Ward limit-cycle trough overshoots (21/49 → 0/49 in the 7×7 test);
# default neg floor (-1e-3) preserved as a backstop. F_N is swept via
# Forcing__fn_scale (a scalar multiplier on fn_monthly, applied before the Fourier
# fit — zero hot-path overhead, see SeasonalForcing in baseline_r0_seasonal_comps).
# fn_scale defaults to 1.0 (identity) so this setup behaves identically to
# setup_baseline_seasonal_slim until fn_scale is overridden at scan time.
setup_baseline_seasonal_slim_tight = xso.setup(
    solver='solve_ivp', model=model_baseline_seasonal, time=seasonal_time,
    input_vars=make_seasonal_input_vars(_PLACEHOLDER_FN_MONTHLY,
                                        _PLACEHOLDER_DE_MONTHLY, _PLACEHOLDER_T_MONTHLY,
                                        n_harmonics=N_HARMONICS),
    output_vars=SLIM_OUTPUT_VARS,
    solver_kwargs={'method': 'RK45', 'atol': 1e-9, 'rtol': 1e-6,
                   'max_step': 1.0, 'instability_neg_threshold': -1e-3})


# =============================================================================
# Setups — ROUTED variant (TEST CONFIG, 2026-06-28). Mirror the two existing
# seasonal setups exactly (same time axis, output vars, solver kwargs); only the
# model + input-vars builder differ. Run side-by-side with
# setup_baseline_seasonal_slim[_tight] to isolate the loss-fate-routing effect.
# =============================================================================
setup_baseline_seasonal_routed_slim = xso.setup(
    solver='solve_ivp', model=model_baseline_seasonal_routed, time=seasonal_time,
    input_vars=make_seasonal_input_vars_routed(_PLACEHOLDER_FN_MONTHLY,
                                               _PLACEHOLDER_DE_MONTHLY, _PLACEHOLDER_T_MONTHLY,
                                               n_harmonics=N_HARMONICS),
    output_vars=SLIM_OUTPUT_VARS, solver_kwargs=IVP_SOLVER_KWARGS)

setup_baseline_seasonal_routed_slim_tight = xso.setup(
    solver='solve_ivp', model=model_baseline_seasonal_routed, time=seasonal_time,
    input_vars=make_seasonal_input_vars_routed(_PLACEHOLDER_FN_MONTHLY,
                                               _PLACEHOLDER_DE_MONTHLY, _PLACEHOLDER_T_MONTHLY,
                                               n_harmonics=N_HARMONICS),
    output_vars=SLIM_OUTPUT_VARS,
    solver_kwargs={'method': 'RK45', 'atol': 1e-9, 'rtol': 1e-6,
                   'max_step': 1.0, 'instability_neg_threshold': -1e-3})
