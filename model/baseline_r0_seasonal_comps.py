"""
baseline_r0_seasonal_comps.py — time-varying (seasonal) forcing components for the
MS3 Cariaco baseline, for the SS-vs-seasonal dynamics test (2026-06-22).

ADDITIVE to baseline_r0_comps.py — nothing in the steady-state components changes.
Three new components, using only the proven XSO foreign-forcing pattern (a forcing
is published by one component and foreign-referenced by its consumers — exactly how
`temperature` already reaches MonodGrowth_T / DistributedGrazing_TypeIII_T):

- SeasonalForcing : a pure forcing provider (no flux). Owns the seasonal F_N(t) as a
  PERIODIC spline through the 12 calendar-month mean F_N values (EMPOWER / Anderson
  et al. 2015 pattern: per=True spline evaluated at np.mod(t, period)), and DERIVES
  d_e(t) and T(t) from F_N(t) via the obs linear regressions (clamped). F_N(t) is the
  SINGLE SOURCE OF TRUTH for the whole seasonal cycle. Publishes three forcings:
  fn (label 'fn'), de (label 'de'), temperature (label 'temperature').
- SeasonalNutrientSupply : the Stock (2008) supply flux J = F_N(t)/d_e(t) -> N,
  foreign-referencing the fn and de forcings.
- DetritusSinking_seasonal : DetritusSinking with d_e read as a foreign FORCING (not a
  parameter), so the seasonal d_e(t) drives the (w_sink/d_e)·D export too — keeping
  d_e the single source of truth across supply and sinking.

Growth and grazing are UNCHANGED: they already foreign-reference the 'temperature'
forcing, now supplied by SeasonalForcing instead of ConstantTemperatureForcing.

Forcing-construction pattern verbatim from the EMPOWER XSO implementation
(forcings.py, StationForcingFromFile.read_intrp_forcing): scipy splrep(..., per=True)
through mid-month day positions with a wrap boundary, evaluated at np.mod(t, period).
"""

import numpy as np
import xso
import scipy.interpolate as intrp


# Mid-month day-of-year positions for a 12-point monthly climatology (EMPOWER).
_DPM = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=float)
_MID_MONTH_DOY = np.cumsum(_DPM) - _DPM / 2.0     # 15.5, 45.0, ..., 350.5


def _build_fn_func(fn_monthly, period, spline_k, spline_s):
    """Periodic spline F_N(t) through the 12 monthly means (EMPOWER pattern), floored
    at 0 to guard against spline undershoot. MODULE-LEVEL (not a component method):
    XSO only carries registered setup_func methods onto the rebuilt component, so the
    forcing setup_funcs call this directly rather than via self."""
    fn_monthly = np.asarray(fn_monthly, dtype=float)
    per = float(period)
    x = np.concatenate([[0.0], _MID_MONTH_DOY, [per]])
    wrap = (fn_monthly[0] + fn_monthly[-1]) / 2.0
    y = np.concatenate([[wrap], fn_monthly, [wrap]])
    tck = intrp.splrep(x, y, per=True, k=int(spline_k), s=float(spline_s))

    def fn_of_t(t):
        return np.maximum(intrp.splev(np.mod(t, per), tck, der=0), 0.0)
    return fn_of_t


@xso.component
class SeasonalForcing:
    """Seasonal forcing provider: F_N(t) periodic spline; d_e(t), T(t) derived.

        F_N(t) = periodic spline through the 12 calendar-month means (>= 0)
        d_e(t) = clip(de_a - de_b * F_N(t), de_lo, de_hi)
        T(t)   = clip(t_a  - t_b  * F_N(t), t_lo,  t_hi)

    Pure forcing component (no flux / no state variable). d_e(t) and T(t) derive from
    F_N(t), so the cycle has ONE source of truth. The three forcings are published
    under labels (fn_label / de_label / temperature_label) for SeasonalNutrientSupply,
    DetritusSinking_seasonal, and MonodGrowth_T / DistributedGrazing_TypeIII_T to
    foreign-reference.
    """
    month = xso.index(dims='month', as_parameter=True,
                      description='calendar-month index for the F_N climatology')
    fn_monthly = xso.parameter(dims='month',
                               description='12 calendar-month mean F_N [mmol N m-2 d-1]')

    period   = xso.parameter(description='forcing period [d] (365)')
    spline_k = xso.parameter(description='periodic spline degree (1 linear / 3 cubic)')
    spline_s = xso.parameter(description='periodic spline smoothing s (0 = interpolate)')

    de_a  = xso.parameter(description='d_e(F_N) intercept [m]')
    de_b  = xso.parameter(description='d_e(F_N) slope [m per mmol N m-2 d-1]')
    de_lo = xso.parameter(description='d_e lower clamp [m]')
    de_hi = xso.parameter(description='d_e upper clamp [m]')

    t_a  = xso.parameter(description='T(F_N) intercept [°C]')
    t_b  = xso.parameter(description='T(F_N) slope [°C per mmol N m-2 d-1]')
    t_lo = xso.parameter(description='T lower clamp [°C]')
    t_hi = xso.parameter(description='T upper clamp [°C]')

    fn          = xso.forcing(setup_func='make_fn',
                              description='seasonal new-N flux F_N(t) [mmol N m-2 d-1]')
    de          = xso.forcing(setup_func='make_de',
                              description='seasonal box depth d_e(t) [m] (single source of truth)')
    temperature = xso.forcing(setup_func='make_temperature',
                              description='seasonal box temperature T(t) [°C]')

    def make_fn(self, fn_monthly, period, spline_k, spline_s):
        return _build_fn_func(fn_monthly, period, spline_k, spline_s)

    def make_de(self, fn_monthly, period, spline_k, spline_s, de_a, de_b, de_lo, de_hi):
        fn_of_t = _build_fn_func(fn_monthly, period, spline_k, spline_s)

        def de_of_t(t):
            return np.clip(de_a - de_b * fn_of_t(t), de_lo, de_hi)
        return de_of_t

    def make_temperature(self, fn_monthly, period, spline_k, spline_s, t_a, t_b, t_lo, t_hi):
        fn_of_t = _build_fn_func(fn_monthly, period, spline_k, spline_s)

        def t_of_t(t):
            return np.clip(t_a - t_b * fn_of_t(t), t_lo, t_hi)
        return t_of_t


@xso.component
class SeasonalNutrientSupply:
    """New-nutrient supply with time-varying forcings, Stock (2008) Eq. 7:
    J(t) = F_N(t) / d_e(t)  [mmol N m-3 d-1]. F_N(t) and d_e(t) are foreign-
    referenced from SeasonalForcing (single source of truth)."""
    var = xso.variable(foreign=True, flux='input', negative=False,
                       description='nutrient pool receiving new-N supply')
    fn = xso.forcing(foreign=True, description='seasonal F_N(t) [mmol N m-2 d-1]')
    de = xso.forcing(foreign=True, description='seasonal d_e(t) [m]')

    @xso.flux
    def input(self, var, fn, de):
        return fn / de


@xso.component
class DetritusSinking_seasonal:
    """DetritusSinking with d_e as a foreign FORCING (seasonal d_e(t)).

        (w_sink / d_e(t)) · D  -> export

    Identical to baseline_r0_comps.DetritusSinking except `de` is read from the
    seasonal d_e(t) forcing rather than a broadcast parameter, keeping d_e the single
    source of truth across supply and sinking."""
    detritus = xso.variable(foreign=True, flux='sinking', negative=True)
    w_sink = xso.parameter(description='detritus sinking velocity w_sink [m d-1]')
    de = xso.forcing(foreign=True,
                     description='seasonal box depth d_e(t) [m] (shared from SeasonalForcing)')

    @xso.flux
    def sinking(self, detritus, w_sink, de):
        return (w_sink / de) * detritus
