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
    DetritusRemineralization, FishGrazing_Kernel_rate,
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
# Input-vars builder
# =============================================================================
def make_seasonal_input_vars(fn_monthly, de_monthly, t_monthly, fish_rate=FISH_RATE,
                             mu_max=mu_max_arr, halfsat=ks_arr,
                             mP=M_P, m_Z=M_Z_BULK,
                             period=PERIOD, n_harmonics=N_HARMONICS):
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
    }
    iv['Inflow'] = {'var': 'N', 'fn': 'fn', 'de': 'de'}
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
