"""
baseline_r0_seasonal_comps.py — time-varying (seasonal) forcing components for the
MS3 Cariaco baseline, for the SS-vs-seasonal dynamics test (2026-06-22).

ADDITIVE to baseline_r0_comps.py — nothing in the steady-state components changes.
Three new components, using only the proven XSO foreign-forcing pattern (a forcing
is published by one component and foreign-referenced by its consumers — exactly how
`temperature` already reaches MonodGrowth_T / DistributedGrazing_TypeIII_T):

- SeasonalForcing : a pure forcing provider (no flux). Owns the seasonal F_N(t), d_e(t)
  and T(t) as INDEPENDENT truncated Fourier fits (n_harmonics=2: annual + semi-annual
  cycle), each through its own 12 calendar-month obs means. Replaced the EMPOWER-style
  periodic cubic spline (2026-06-24) to eliminate Runge-type wiggle overshoot; standard
  in physical oceanography. d_e(t) and T(t) are forced DIRECTLY from the obs era
  climatologies (no longer derived from F_N; 2026-06-22). Publishes three forcings:
  fn (label 'fn'), de (label 'de'), temperature (label 'temperature').
- SeasonalNutrientSupply : the Stock (2008) supply flux J = F_N(t)/d_e(t) -> N,
  foreign-referencing the fn and de forcings.
- DetritusSinking_seasonal : DetritusSinking with d_e read as a foreign FORCING (not a
  parameter), so the seasonal d_e(t) drives the (w_sink/d_e)·D export too — keeping
  d_e the single source of truth across supply and sinking.

Growth and grazing are UNCHANGED: they already foreign-reference the 'temperature'
forcing, now supplied by SeasonalForcing instead of ConstantTemperatureForcing.

Forcing construction: truncated Fourier fit (default n_harmonics=2) at the 12 mid-month
DOY positions, evaluated via np.mod(t, period). Replaced the EMPOWER-style periodic cubic
spline (2026-06-24) which suffered Runge-type overshoot between monthly points.
The legacy _build_fn_func is retained for diagnostic comparisons.
"""

import numpy as np
import xso
import scipy.interpolate as intrp


# Mid-month day-of-year positions for a 12-point monthly climatology (EMPOWER).
_DPM = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=float)
_MID_MONTH_DOY = np.cumsum(_DPM) - _DPM / 2.0     # 15.5, 45.0, ..., 350.5


def _build_fn_func(fn_monthly, period, spline_k, spline_s):
    """LEGACY — periodic cubic spline F_N(t) through 12 monthly means.
    Retained for reference / diagnostic comparisons. Production forcing now uses
    _build_fourier_func (2-harmonic Fourier fit; 2026-06-24)."""
    fn_monthly = np.asarray(fn_monthly, dtype=float)
    per = float(period)
    x = np.concatenate([[0.0], _MID_MONTH_DOY, [per]])
    wrap = (fn_monthly[0] + fn_monthly[-1]) / 2.0
    y = np.concatenate([[wrap], fn_monthly, [wrap]])
    tck = intrp.splrep(x, y, per=True, k=int(spline_k), s=float(spline_s))

    def fn_of_t(t):
        return np.maximum(intrp.splev(np.mod(t, per), tck, der=0), 0.0)
    return fn_of_t


def _build_fourier_func(monthly, period, n_harmonics=2):
    """Truncated Fourier (harmonic) fit through 12 monthly values, evaluated as a
    periodic function of continuous time.  Replaces _build_fn_func (2026-06-24).

        y(t) = a_0 + Σ_{k=1}^{n} [ a_k cos(2πkt/T) + b_k sin(2πkt/T) ]

    Least-squares fit at the 12 mid-month DOY positions.  Inherently smooth and
    periodic — no Runge-type overshoot, no free smoothing parameter.  Standard in
    physical oceanography (annual + semi-annual = 2 harmonics).  Floored at 0 for
    variables that must be non-negative (F_N, d_e).

    MODULE-LEVEL (not a component method): XSO only carries registered setup_func
    methods onto the rebuilt component, so the forcing setup_funcs call this directly.
    """
    monthly = np.asarray(monthly, dtype=float)
    per = float(period)
    n = int(n_harmonics)
    t_pts = _MID_MONTH_DOY
    ncols = 1 + 2 * n
    A = np.ones((12, ncols))
    for k in range(1, n + 1):
        A[:, 2*k - 1] = np.cos(2 * np.pi * k * t_pts / per)
        A[:, 2*k]     = np.sin(2 * np.pi * k * t_pts / per)
    coeffs, _, _, _ = np.linalg.lstsq(A, monthly, rcond=None)

    @np.vectorize
    def fn_of_t(t):
        t_mod = t % per
        val = coeffs[0]
        for k in range(1, n + 1):
            val += coeffs[2*k - 1] * np.cos(2 * np.pi * k * t_mod / per)
            val += coeffs[2*k]     * np.sin(2 * np.pi * k * t_mod / per)
        return max(val, 0.0)
    return fn_of_t


@xso.component
class SeasonalForcing:
    """Seasonal forcing provider: F_N(t), d_e(t), T(t) as independent Fourier fits.

        F_N(t) = 2-harmonic Fourier fit through the 12 calendar-month F_N obs means (>= 0)
        d_e(t) = 2-harmonic Fourier fit through the 12 calendar-month d_e obs means (>= 0)
        T(t)   = 2-harmonic Fourier fit through the 12 calendar-month T  obs means

    Pure forcing component (no flux / no state variable). Replaced periodic cubic spline
    with truncated Fourier fit (2026-06-24) to eliminate Runge-type overshoot. d_e(t) and
    T(t) are forced DIRECTLY from their obs climatologies (not derived from F_N; 2026-06-22).
    The three forcings are published under labels (fn_label / de_label / temperature_label)
    for SeasonalNutrientSupply, DetritusSinking_seasonal, and MonodGrowth_T /
    DistributedGrazing_TypeIII_T to foreign-reference.
    """
    month = xso.index(dims='month', as_parameter=True,
                      description='calendar-month index for the F_N climatology')
    fn_monthly = xso.parameter(dims='month',
                               description='12 calendar-month mean F_N [mmol N m-2 d-1]')

    period       = xso.parameter(description='forcing period [d] (365)')
    n_harmonics  = xso.parameter(description='number of Fourier harmonics (default 2 = annual + semi-annual)')

    # Scalar F_N multiplier for parameter scans (added 2026-06-25). Applied to
    # fn_monthly BEFORE the Fourier fit -- because the fit is linear in the data,
    # Fourier(c*monthly) = c*Fourier(monthly), so the coefficients carry the
    # scaling and the forcing closure (called per solver step) has zero hot-path
    # overhead. Default 1.0 = identity (existing behaviour).
    fn_scale = xso.parameter(
        description='F_N forcing scale multiplier [dimensionless]; multiplies '
                    'fn_monthly before the Fourier fit (zero hot-path cost). '
                    'Default 1.0; sweep via SeasonalForcing__fn_scale for 2D parscans.')

    de_monthly = xso.parameter(dims='month',
                               description='12 calendar-month mean d_e [m] (forced directly from obs)')
    t_monthly  = xso.parameter(dims='month',
                               description='12 calendar-month mean T [°C] (forced directly from obs)')

    fn          = xso.forcing(setup_func='make_fn',
                              description='seasonal new-N flux F_N(t) [mmol N m-2 d-1]')
    de          = xso.forcing(setup_func='make_de',
                              description='seasonal box depth d_e(t) [m] (single source of truth)')
    temperature = xso.forcing(setup_func='make_temperature',
                              description='seasonal box temperature T(t) [°C]')

    def make_fn(self, fn_monthly, period, n_harmonics, fn_scale):
        return _build_fourier_func(np.asarray(fn_monthly) * float(fn_scale),
                                   period, n_harmonics)

    def make_de(self, de_monthly, period, n_harmonics):
        return _build_fourier_func(de_monthly, period, n_harmonics)

    def make_temperature(self, t_monthly, period, n_harmonics):
        return _build_fourier_func(t_monthly, period, n_harmonics)


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
