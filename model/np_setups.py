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

# IVP run duration. Reduced 2026-05-18 from 5000 d to 2000 d: the N-P
# competitive-exclusion dynamics are well-resolved by ~1500 d (losing
# classes 2-3 e-folds below their initial value), and integrating the
# asymptotic-zero regime past that is slow (stiff near-zero behaviour).
IVP_TIME_END   = 2000
ivp_time_array = np.arange(0, IVP_TIME_END, 1)


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
#   const — size-independent, Taniguchi value 0.0015 d⁻¹
#   allom — Λ(s) = Λ₀ · s^(-0.25), the MTE prediction Taniguchi explicitly
#           rejected; included here so we can show that even with this
#           classic size-dependence the system still collapses to a single
#           surviving class. Anchored so Λ(1 µm) = the const value.
LAMBDA_CONST     = 0.0015
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
# MODEL ASSEMBLY — CLOSED VARIANT
# =============================================================================
model_closed = xso.create({
    'Nutrient':      Nutrient,
    'Phytoplankton': PhytoSizeSpectrum,
    'Growth':        MonodGrowth_NP,
    'PhytoLoss':     PhytoLinearLoss_recycled,
})

# Shared structural input_vars (the dict that doesn't change between
# const and allom variants of the closed model)
_closed_base = {
    'Nutrient': {
        'value_label': 'N',
        'value_init':  N_INIT_CLOSED,
    },
    'Phytoplankton': {
        'biomass_label':   'P',
        'biomass_init':    phyto_init,
        'phyto_esd_index': phyto_esd.tolist(),
        'phyto_esd_label': 'phyto_esd',
    },
    'Growth': {
        'resource': 'N',
        'consumer': 'P',
        'mu_max':   mu_max_arr,
        'halfsat':  k_s_arr,
    },
    # 'PhytoLoss' filled in per variant below
}


def _closed_inputs(loss_rate):
    return {
        **_closed_base,
        'PhytoLoss': {
            'population': 'P',
            'nutrient':   'N',
            'rate':       loss_rate,
        },
    }


# IVP + stability setups, closed-const
model_setup_closed_const = xso.setup(
    solver='solve_ivp', model=model_closed,
    time=ivp_time_array,
    input_vars=_closed_inputs(lambda_arr_const),
)
model_setup_closed_const_stability = xso.setup(
    solver='stability', model=model_closed,
    time=[0, 1],
    input_vars=_closed_inputs(lambda_arr_const),
)

# IVP + stability setups, closed-allom
model_setup_closed_allom = xso.setup(
    solver='solve_ivp', model=model_closed,
    time=ivp_time_array,
    input_vars=_closed_inputs(lambda_arr_allom),
)
model_setup_closed_allom_stability = xso.setup(
    solver='stability', model=model_closed,
    time=[0, 1],
    input_vars=_closed_inputs(lambda_arr_allom),
)


# =============================================================================
# MODEL ASSEMBLY — OPEN VARIANT (CHEMOSTAT)
# =============================================================================
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

_open_base = {
    'Nutrient': {
        'value_label': 'N',
        'value_init':  N_INIT_OPEN,
    },
    'Phytoplankton': {
        'biomass_label':   'P',
        'biomass_init':    phyto_init,
        'phyto_esd_index': phyto_esd.tolist(),
        'phyto_esd_label': 'phyto_esd',
    },
    'Growth': {
        'resource': 'N',
        'consumer': 'P',
        'mu_max':   mu_max_arr,
        'halfsat':  k_s_arr,
    },
    # 'PhytoLoss' per variant
    'FN_Forcing': {
        'forcing_label': 'FN_supply',
        'value':         F_N_DEFAULT,
    },
    'FN_Input': {
        'var':     'N',
        'forcing': 'FN_supply',
        'rate':    1.0 / D_E,    # gives flux = F_N / d_e
    },
    'DilutionN': {
        'var':  'N',
        'rate': DILUTION_RATE,
    },
    'DilutionP': {
        'var':  'P',
        'rate': DILUTION_RATE,
    },
}


def _open_inputs(loss_rate):
    return {
        **_open_base,
        'PhytoLoss': {
            'population': 'P',
            'nutrient':   'N',
            'rate':       loss_rate,
        },
    }


# IVP + stability setups, open-const
model_setup_open_const = xso.setup(
    solver='solve_ivp', model=model_open,
    time=ivp_time_array,
    input_vars=_open_inputs(lambda_arr_const),
)
model_setup_open_const_stability = xso.setup(
    solver='stability', model=model_open,
    time=[0, 1],
    input_vars=_open_inputs(lambda_arr_const),
)

# IVP + stability setups, open-allom
model_setup_open_allom = xso.setup(
    solver='solve_ivp', model=model_open,
    time=ivp_time_array,
    input_vars=_open_inputs(lambda_arr_allom),
)
model_setup_open_allom_stability = xso.setup(
    solver='stability', model=model_open,
    time=[0, 1],
    input_vars=_open_inputs(lambda_arr_allom),
)
