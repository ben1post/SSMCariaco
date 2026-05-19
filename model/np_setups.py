"""
N–P-only spectral model — XSO setups
====================================
Four model variants for the Step 1 diagnostic (MS3 layered construction
plan, 2026-05-18). Each variant exposes both an IVP setup (`solve_ivp`,
5000 d) and a stability setup (`fsolve` Jacobian eigenvalue analysis,
length-2 time array) for the IVP-then-stability parscan workflow used
in `run_1d_scan_spectrum.py`.

Variants
--------
* `closed_const` — closed (N_T conserved), size-independent Λ
* `closed_allom` — closed,                size-dependent Λ(s) = Λ₀·s^(-0.25)
* `open_const`   — open chemostat (F_N supply + λ-dilution on N & P),
                   size-independent Λ
* `open_allom`   — open chemostat,        size-dependent Λ

Module-level exports
--------------------
* Models:           `model_closed`, `model_open`
* IVP setups:       `model_setup_closed_const`, `model_setup_closed_allom`,
                    `model_setup_open_const`,   `model_setup_open_allom`
* Stability setups: same names with `_stability` suffix
* Helpers:          `phyto_esd`, `n_classes`, `generate_size_classes`,
                    `avg_tail` (re-export)

Scan axes
---------
* Closed variants: `'Nutrient__value_init'` (sets N_T at t = 0; closed
  dynamics preserve `N_T = N + ΣP` thereafter — see MS3 Background §2026-05-18).
* Open variants:   `'FN_Forcing__value'` (the F_N supply rate; at steady
  state `N_T = F_N / (λ · d_e)` per Eq. F-N-Nstar-bridge in the LaTeX).

Size grid
---------
40 log-spaced classes from 0.2 µm to 200 µm (Sieburth-compatible range,
~13 classes per decade — Banas 2011 ballpark per Benny's pref). Change
via the module-level constant `N_CLASSES` (or call
`generate_size_classes(n=...)` directly).
"""

import numpy as np
import xso

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from xso.parscans import avg_tail  # re-export for parscan worker discovery

from np_comps import (
    Nutrient,
    PhytoSizeSpectrum,
    ConstantExternalNutrient,
    LinearForcingInput,
    MonodGrowth_NP,
    PhytoLinearLoss_recycled,
    ChemostatDilution_Scalar,
    ChemostatDilution_PhytoDim,
)


# =============================================================================
# TOP-LEVEL GRID / PARAMETER CONSTANTS
# =============================================================================
N_CLASSES = 40
ESD_MIN   = 0.2
ESD_MAX   = 200.0

# IVP run duration. Restored 2026-05-18 from 2000 d to 5000 d after the
# LSODA + relaxed-atol switch (see IVP_SOLVER_KWARGS below) made each
# cell sub-second; 2000 d was a workaround for RK45 stiffness slowness
# that is no longer relevant. At 5000 d the slowest-decaying loser
# classes have 5+ e-folds below their initial value, so the tail-mean
# represents a more fully resolved steady state.
IVP_TIME_END   = 5000
ivp_time_array = np.arange(0, IVP_TIME_END, 1)

# scipy.integrate.solve_ivp method + tolerances (passed via XSO's
# solver_kwargs hook added 2026-05-18).
#
# LSODA auto-switches between Adams (non-stiff) and BDF (stiff) and
# is the safe default for size-spectrum systems with mixed regimes.
#
# Tolerances loosened from XSO's RK45 defaults (atol=1e-9, rtol=1e-6).
# In competitive-exclusion N-P dynamics, ~40 loser classes spend long
# stretches at <1e-6 mmol N m-3 — biologically noise — and tight atol
# forces the step controller to spend most of its budget tracking
# those values faithfully. The relaxed values match the physically
# meaningful precision: any concentration below ~1e-6 is rounding,
# and 1e-4 relative is finer than every comparison we make in the
# manuscript metrics. Final-state comparison after this change should
# still match RK45/default-tol within fractions of a percent.
IVP_SOLVER_KWARGS = {'method': 'LSODA', 'atol': 1e-6, 'rtol': 1e-4}


def generate_size_classes(n=None, esd_min=None, esd_max=None):
    """Log-spaced phyto ESD grid. Defaults pulled from module-level constants."""
    if n is None:        n        = N_CLASSES
    if esd_min is None:  esd_min  = ESD_MIN
    if esd_max is None:  esd_max  = ESD_MAX
    return np.logspace(np.log10(esd_min), np.log10(esd_max), n)


phyto_esd = generate_size_classes()
n_classes = len(phyto_esd)


# =============================================================================
# ALLOMETRIES — Taniguchi 2014 Table 1 (verbatim for μ, k_s)
# =============================================================================
mu_max_arr = 1.36 * phyto_esd ** (-0.16)
k_s_arr    = 0.33 * phyto_esd ** ( 0.48)

# Two Λ variants:
#   const — size-independent
#   allom — Λ(s) = Λ₀ · s^(-0.25), the MTE prediction Taniguchi explicitly
#           rejected; included here to show that even with this classic
#           size-dependence the system still collapses to a single surviving
#           class. Anchored so Λ(1 µm) = the const value.
#
# 2026-05-18: prefactor raised from Taniguchi's 0.0015 d⁻¹ to a more
# conventional 0.1 d⁻¹ (Stock 2008 background phyto mortality; Banas
# 2011 m_P ≈ 0.1·μ). The Taniguchi value produces decay timescales of
# ~20 000 d for classes near the competitive-exclusion winner, far past
# our 2000 d IVP window; with 0.1 d⁻¹ adjacent-class decay timescales
# are ~330 d so loser classes are 99 %+ decayed by t = 2000. Qualitative
# competitive-exclusion result is unchanged; visualisation becomes clean.
LAMBDA_CONST     = 0.1
LAMBDA_ALLOM_EXP = -0.25
lambda_arr_const = np.full(n_classes, LAMBDA_CONST)
lambda_arr_allom = LAMBDA_CONST * phyto_esd ** LAMBDA_ALLOM_EXP


# =============================================================================
# OPEN-VARIANT PARAMETERS — F_N supply + chemostat dilution
# =============================================================================
D_E              = 50.0     # m, surface-box depth
DILUTION_RATE    = 0.05     # d⁻¹, single λ applied to both N and P
F_N_DEFAULT      = 5.0      # mmol N m⁻² d⁻¹ (scanned at run time)


# =============================================================================
# INITIAL CONDITIONS
# =============================================================================
P_INIT_PER_CLASS = 1e-3                 # mmol N m⁻³, small uniform seed
phyto_init       = np.full(n_classes, P_INIT_PER_CLASS)
N_INIT_CLOSED    = 15.0                 # closed: this sets N_T; scanned at run time
N_INIT_OPEN      = 1.0                  # open: irrelevant to SS, just a seed


# =============================================================================
# MODELS
# =============================================================================
model_closed = xso.create({
    'Nutrient':      Nutrient,
    'Phytoplankton': PhytoSizeSpectrum,
    'Growth':        MonodGrowth_NP,
    'PhytoLoss':     PhytoLinearLoss_recycled,
})

model_open = xso.create({
    'Nutrient':      Nutrient,
    'Phytoplankton': PhytoSizeSpectrum,
    'Growth':        MonodGrowth_NP,
    'PhytoLoss':     PhytoLinearLoss_recycled,
    'FN_Forcing':    ConstantExternalNutrient,
    'FN_Input':      LinearForcingInput,
    'DilutionN':     ChemostatDilution_Scalar,
    'DilutionP':     ChemostatDilution_PhytoDim,
})


# =============================================================================
# SETUP — CLOSED, SIZE-INDEPENDENT Λ
# =============================================================================
model_setup_closed_const = xso.setup(
    solver='solve_ivp', model=model_closed,
    time=ivp_time_array,
    input_vars={
        'Nutrient':      {'value_label': 'N', 'value_init': N_INIT_CLOSED},
        'Phytoplankton': {'biomass_label': 'P', 'biomass_init': phyto_init,
                          'phyto_esd_index': phyto_esd.tolist(),
                          'phyto_esd_label': 'phyto_esd'},
        'Growth':        {'resource': 'N', 'consumer': 'P',
                          'mu_max': mu_max_arr, 'halfsat': k_s_arr},
        'PhytoLoss':     {'population': 'P', 'nutrient': 'N',
                          'rate': lambda_arr_const},
    },
    solver_kwargs=IVP_SOLVER_KWARGS,
)

model_setup_closed_const_stability = xso.setup(
    solver='stability', model=model_closed,
    time=[0, 1],
    input_vars={
        'Nutrient':      {'value_label': 'N', 'value_init': N_INIT_CLOSED},
        'Phytoplankton': {'biomass_label': 'P', 'biomass_init': phyto_init,
                          'phyto_esd_index': phyto_esd.tolist(),
                          'phyto_esd_label': 'phyto_esd'},
        'Growth':        {'resource': 'N', 'consumer': 'P',
                          'mu_max': mu_max_arr, 'halfsat': k_s_arr},
        'PhytoLoss':     {'population': 'P', 'nutrient': 'N',
                          'rate': lambda_arr_const},
    },
)


# =============================================================================
# SETUP — CLOSED, SIZE-DEPENDENT Λ(s) = Λ₀ · s^(-0.25)
# =============================================================================
model_setup_closed_allom = xso.setup(
    solver='solve_ivp', model=model_closed,
    time=ivp_time_array,
    input_vars={
        'Nutrient':      {'value_label': 'N', 'value_init': N_INIT_CLOSED},
        'Phytoplankton': {'biomass_label': 'P', 'biomass_init': phyto_init,
                          'phyto_esd_index': phyto_esd.tolist(),
                          'phyto_esd_label': 'phyto_esd'},
        'Growth':        {'resource': 'N', 'consumer': 'P',
                          'mu_max': mu_max_arr, 'halfsat': k_s_arr},
        'PhytoLoss':     {'population': 'P', 'nutrient': 'N',
                          'rate': lambda_arr_allom},
    },
    solver_kwargs=IVP_SOLVER_KWARGS,
)

model_setup_closed_allom_stability = xso.setup(
    solver='stability', model=model_closed,
    time=[0, 1],
    input_vars={
        'Nutrient':      {'value_label': 'N', 'value_init': N_INIT_CLOSED},
        'Phytoplankton': {'biomass_label': 'P', 'biomass_init': phyto_init,
                          'phyto_esd_index': phyto_esd.tolist(),
                          'phyto_esd_label': 'phyto_esd'},
        'Growth':        {'resource': 'N', 'consumer': 'P',
                          'mu_max': mu_max_arr, 'halfsat': k_s_arr},
        'PhytoLoss':     {'population': 'P', 'nutrient': 'N',
                          'rate': lambda_arr_allom},
    },
)


# =============================================================================
# SETUP — OPEN CHEMOSTAT, SIZE-INDEPENDENT Λ
# =============================================================================
model_setup_open_const = xso.setup(
    solver='solve_ivp', model=model_open,
    time=ivp_time_array,
    input_vars={
        'Nutrient':      {'value_label': 'N', 'value_init': N_INIT_OPEN},
        'Phytoplankton': {'biomass_label': 'P', 'biomass_init': phyto_init,
                          'phyto_esd_index': phyto_esd.tolist(),
                          'phyto_esd_label': 'phyto_esd'},
        'Growth':        {'resource': 'N', 'consumer': 'P',
                          'mu_max': mu_max_arr, 'halfsat': k_s_arr},
        'PhytoLoss':     {'population': 'P', 'nutrient': 'N',
                          'rate': lambda_arr_const},
        'FN_Forcing':    {'forcing_label': 'FN_supply', 'value': F_N_DEFAULT},
        'FN_Input':      {'var': 'N', 'forcing': 'FN_supply',
                          'rate': 1.0 / D_E},
        'DilutionN':     {'var': 'N', 'rate': DILUTION_RATE},
        'DilutionP':     {'var': 'P', 'rate': DILUTION_RATE},
    },
    solver_kwargs=IVP_SOLVER_KWARGS,
)

model_setup_open_const_stability = xso.setup(
    solver='stability', model=model_open,
    time=[0, 1],
    input_vars={
        'Nutrient':      {'value_label': 'N', 'value_init': N_INIT_OPEN},
        'Phytoplankton': {'biomass_label': 'P', 'biomass_init': phyto_init,
                          'phyto_esd_index': phyto_esd.tolist(),
                          'phyto_esd_label': 'phyto_esd'},
        'Growth':        {'resource': 'N', 'consumer': 'P',
                          'mu_max': mu_max_arr, 'halfsat': k_s_arr},
        'PhytoLoss':     {'population': 'P', 'nutrient': 'N',
                          'rate': lambda_arr_const},
        'FN_Forcing':    {'forcing_label': 'FN_supply', 'value': F_N_DEFAULT},
        'FN_Input':      {'var': 'N', 'forcing': 'FN_supply',
                          'rate': 1.0 / D_E},
        'DilutionN':     {'var': 'N', 'rate': DILUTION_RATE},
        'DilutionP':     {'var': 'P', 'rate': DILUTION_RATE},
    },
)


# =============================================================================
# SETUP — OPEN CHEMOSTAT, SIZE-DEPENDENT Λ(s) = Λ₀ · s^(-0.25)
# =============================================================================
model_setup_open_allom = xso.setup(
    solver='solve_ivp', model=model_open,
    time=ivp_time_array,
    input_vars={
        'Nutrient':      {'value_label': 'N', 'value_init': N_INIT_OPEN},
        'Phytoplankton': {'biomass_label': 'P', 'biomass_init': phyto_init,
                          'phyto_esd_index': phyto_esd.tolist(),
                          'phyto_esd_label': 'phyto_esd'},
        'Growth':        {'resource': 'N', 'consumer': 'P',
                          'mu_max': mu_max_arr, 'halfsat': k_s_arr},
        'PhytoLoss':     {'population': 'P', 'nutrient': 'N',
                          'rate': lambda_arr_allom},
        'FN_Forcing':    {'forcing_label': 'FN_supply', 'value': F_N_DEFAULT},
        'FN_Input':      {'var': 'N', 'forcing': 'FN_supply',
                          'rate': 1.0 / D_E},
        'DilutionN':     {'var': 'N', 'rate': DILUTION_RATE},
        'DilutionP':     {'var': 'P', 'rate': DILUTION_RATE},
    },
    solver_kwargs=IVP_SOLVER_KWARGS,
)

model_setup_open_allom_stability = xso.setup(
    solver='stability', model=model_open,
    time=[0, 1],
    input_vars={
        'Nutrient':      {'value_label': 'N', 'value_init': N_INIT_OPEN},
        'Phytoplankton': {'biomass_label': 'P', 'biomass_init': phyto_init,
                          'phyto_esd_index': phyto_esd.tolist(),
                          'phyto_esd_label': 'phyto_esd'},
        'Growth':        {'resource': 'N', 'consumer': 'P',
                          'mu_max': mu_max_arr, 'halfsat': k_s_arr},
        'PhytoLoss':     {'population': 'P', 'nutrient': 'N',
                          'rate': lambda_arr_allom},
        'FN_Forcing':    {'forcing_label': 'FN_supply', 'value': F_N_DEFAULT},
        'FN_Input':      {'var': 'N', 'forcing': 'FN_supply',
                          'rate': 1.0 / D_E},
        'DilutionN':     {'var': 'N', 'rate': DILUTION_RATE},
        'DilutionP':     {'var': 'P', 'rate': DILUTION_RATE},
    },
)
