"""
Taniguchi Model 1 — XSO setup
=============================
Assembles the literature-equivalent NPZ baseline of Taniguchi, Franks &
Poulin (2014, *Mar Ecol Prog Ser* **514**: 13–33), Model 1 (single-prey,
herbivorous, predator–prey size ratio r = 1).

Exports (module-level globals expected by `xso.parscans` workers)
----------------------------------------------------------------
* `model`                   — the xsimlab.Model object (xso.create result)
* `model_setup`             — solve_ivp transient run, full output
* `model_setup_slim`        — solve_ivp transient, reduced output for scans
* `model_setup_stability`   — stability solver, length-2 time array
* `phyto_esd`, `zoo_esd`    — the size grids (n = 501, Taniguchi-verbatim)
* `avg_tail`                — re-exported from xso.parscans for convenience

Companion files
---------------
* `taniguchi_comps.py`                       — component definitions
* `model context/Taniguchi_Model1_Baseline.tex` — analytical baseline
* `model context/Model Equations.md` §8–§14  — equation-set summary
* `code/XSO_HANDOFF.md`                      — XSO framework reference

Parameter values — VERBATIM from Taniguchi 2014 Table 1
-------------------------------------------------------
Allometries are power laws x(s) = x_0 · s^e_x with ESD s in µm:

    μ(s)   = 1.36  · s^(-0.16)   [d-1]       max phyto growth rate
    k_s(s) = 0.33  · s^(+0.48)   [µmol N L-1] phyto half-saturation
    g(s)   = 33.96 · s^(-0.66)   [d-1]       max microzoo grazing rate
    k_z(s) = 17.92 · s^(-0.64)   [µmol N L-1] grazing half-saturation

Size-independent constants:

    Γ = 0.31     gross growth efficiency
    Λ = 0.0015 d-1   phyto biomass-associated loss
    Δ = 0.025  d-1   microzoo biomass-associated loss
    r = 1        predator–prey ESD ratio (Model 1)
    N_T = 15 µmol N L-1   total nitrogen (Taniguchi default reference)

Rates are temperature-corrected to 20 °C using Brown et al. (2004) MTE
with activation energies E = 0.36 eV (growth) and 0.67 eV (grazing).
The implied Q₁₀ from Taniguchi's correction is 1.60–1.64 for phyto
growth and 2.43–2.53 for grazing (Taniguchi pp. 17, 19).

Size grid — VERBATIM formula from Taniguchi 2014 p. 20
------------------------------------------------------
    s_j = 0.8 × 1.0182^j µm,   j = 0, 1, ..., 500   (n = 501 classes)

This formula gives ~128 classes per decade and spans [0.8 µm, ~6.6 mm]
(s_500 = 0.8 × 1.0182^500 ≈ 6.6 × 10³ µm). NB: the source descriptive
text in `Taniguchi_Model1_Baseline.tex` line 50 and `Model Equations.md`
line 286 both render the upper extent as "~65 mm" and the resolution as
"~38 classes per decade" — those descriptions are inconsistent with the
formula they quote on the same line, and worth verifying against the
original paper. The numeric formula is unambiguous and implemented here
verbatim; practical biomass distributions span j = 0..430 only
(s ≤ ~1900 µm), with empty extinct classes above s_max(N_T) —
Taniguchi_Model1_Baseline.tex §5.

Initial conditions
------------------
Small uniform positive seeds across all 501 classes; N_0 is set so
that N_0 + Σ P_init + Σ Z_init = N_T exactly. The system converges
to its analytical steady state regardless of seed shape (the
closure constraint is preserved at all times by the ODE
specification — Λ·P, Δ·Z, (1−Γ)·G all return to N).

Reference predictions to verify against
---------------------------------------
At N_T = 15 µmol N L-1 (Taniguchi_Model1_Baseline.tex Table 1, §6):

    s_max          ≈ 175 µm
    P*(1.0 µm)     ≈ 0.0425 µmol N L-1
    P*(6.3 µm)     ≈ 0.0441 µmol N L-1
    P*(63 µm)      ≈ 0.0462 µmol N L-1
    Z*(s) per class set by Eq. 11 (mixed phyto+grazer allometry)
    Cariaco centroid  ≈ +1.00   (resolved Pico bin convention)
    Cariaco slope_2pt ≈ +0.13

These are the numerical targets for direct verification of the XSO
implementation against the analytical predictions of the LaTeX
document.
"""

import numpy as np
import xso

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from xso.parscans import avg_tail  # re-export for parscan worker discovery

from taniguchi_comps import (
    Nutrient, PhytoSizeSpectrum, ZooSizeSpectrum,
    MonodGrowth_Tani, PhytoLinearLoss_Tani,
    MatchedGrazing_Tani, ZooLinearLoss_Tani,
)


# =============================================================================
# ALLOMETRIC FUNCTIONS — Taniguchi 2014 Table 1, verbatim
# =============================================================================

def compute_mu_max_taniguchi(esd):
    """Maximum phytoplankton growth rate (Taniguchi Eq. 8 family; Table 1, Fig. 3a).

        μ(s) = 1.36 · s^(-0.16)   [d-1]

    Monotonically decreasing in size — small cells grow fastest. r² = 0.12,
    n = 101 in the Taniguchi data compilation. Marañón's unimodal pattern
    is acknowledged in Taniguchi's Discussion (p. 21) but rejected here
    for analytical tractability; swapping in Marañón is one of the
    layered MS3 deviations (Taniguchi_Model1_Baseline.tex §7, deviation 4).
    """
    return 1.36 * esd ** (-0.16)


def compute_k_s_taniguchi(esd):
    """Phytoplankton half-saturation constant (Taniguchi Table 1, Fig. 3b).

        k_s(s) = 0.33 · s^(+0.48)   [µmol N L-1]

    Increases with size — larger cells less efficient at low N. r² = 0.35,
    n = 31. Exponent +0.48 is the *lowest defensible value* in the
    literature; Marañón gives +0.81 (used in MS3-as-built); Eppley +1.0.
    """
    return 0.33 * esd ** (+0.48)


def compute_g_max_taniguchi(esd):
    """Maximum microzooplankton grazing rate (Taniguchi Table 1, Fig. 3c).

        g(s) = 33.96 · s^(-0.66)   [d-1]

    r² = 0.32, n = 46. **The most sensitive parameter in both spectra**
    (Taniguchi Tables 3, 4 — sensitivity ≈ +4.98 for phyto spectrum,
    +519 for microzoo spectrum). MS3 uses Stock/Hansen e_g = -0.48; the
    -0.66 vs -0.48 disagreement is the single most consequential
    allometric mismatch (Model Equations §9.3).
    """
    return 33.96 * esd ** (-0.66)


def compute_k_z_taniguchi(esd):
    """Microzooplankton grazing half-saturation (Taniguchi Table 1, Fig. 3d).

        k_z(s) = 17.92 · s^(-0.64)   [µmol N L-1]

    r² = 0.35, n = 40. **Size-dependent** — the key analytical-slope lever
    in Taniguchi's framework. Hansen 1997 (the underlying data source) did
    not find a significant trend; Taniguchi's k_z trend comes from a
    different subset of the same data with a forced common slope across
    protistan groups. MS3-as-built / Stock / Banas use uniform K_sZ — the
    structural disagreement is flagged in Model Equations §9.4.
    """
    return 17.92 * esd ** (-0.64)


def generate_taniguchi_size_grid(n=501, s0=0.8, ratio=1.0182):
    """Generate Taniguchi's logarithmic size grid verbatim (p. 20).

        s_j = s0 · ratio^j   for j = 0, 1, ..., n-1

    Defaults reproduce Taniguchi's published *formula* exactly: 501
    classes from 0.8 µm to ~6.6 mm (s_500 = 0.8 × 1.0182^500), giving
    ~128 classes per decade. Practical biomass distributions span only
    j = 0..430 (s ≤ ~1900 µm); extinct classes above s_max(N_T) carry
    zero biomass at steady state but are still integrated.
    """
    j = np.arange(n)
    return s0 * ratio ** j


# =============================================================================
# SIZE GRID — Taniguchi 2014 p. 20, verbatim
# =============================================================================
n_classes = 501
phyto_esd = generate_taniguchi_size_grid(n=n_classes)
# For Model 1 (r = 1) phyto and zoo share the size grid numerically.
# Keeping a separate `zoo_esd` reference simplifies the r → 10 deviation
# layer later (Taniguchi_Model1_Baseline.tex §7, deviation 1).
zoo_esd = phyto_esd.copy()


# =============================================================================
# PER-CLASS ALLOMETRIC PARAMETER ARRAYS
# =============================================================================
mu_max_arr  = compute_mu_max_taniguchi(phyto_esd)    # dims='phyto'
k_s_arr     = compute_k_s_taniguchi(phyto_esd)       # dims='phyto'
g_max_arr   = compute_g_max_taniguchi(zoo_esd)       # dims='zoo'
k_z_arr     = compute_k_z_taniguchi(zoo_esd)         # dims='zoo'


# =============================================================================
# SIZE-INDEPENDENT SCALAR PARAMETERS — Taniguchi 2014 Table 1, verbatim
# =============================================================================
gamma_val   = 0.31      # Γ — gross growth efficiency (size-independent)
lambda_val  = 0.0015    # Λ — phyto biomass-associated loss rate [d-1]
delta_val   = 0.025     # Δ — microzoo biomass-associated loss rate [d-1]


# =============================================================================
# TOTAL NITROGEN AND INITIAL CONDITIONS
# =============================================================================
# Total nitrogen — Taniguchi's default reference value (Table 1, p. 20).
# Taniguchi additionally reports runs at N_T ∈ {10, 20, 25, 30}; sweeping
# N_T is the canonical first scan with this baseline.
N_T = 15.0

# Small uniform positive seeds. Picked so that Σ P_init + Σ Z_init ≪ N_T
# — the bulk of nitrogen starts in the dissolved pool and the dynamics
# redistribute it across the spectrum.
P_init_per_class = 1e-4
Z_init_per_class = 1e-4
phyto_init = np.full(n_classes, P_init_per_class)
zoo_init   = np.full(n_classes, Z_init_per_class)

# Set N_0 so that the closure N_T = N + ΣP + ΣZ is satisfied at t = 0.
N_init = N_T - phyto_init.sum() - zoo_init.sum()
assert N_init > 0, (
    f"N_init = {N_init} <= 0 — initial biomass exceeds N_T. "
    "Lower per-class seeds or raise N_T."
)


# =============================================================================
# BUILD MODEL
# =============================================================================
model = xso.create({
    'Nutrient':         Nutrient,
    'Phytoplankton':    PhytoSizeSpectrum,
    'Microzooplankton': ZooSizeSpectrum,
    'Growth':           MonodGrowth_Tani,
    'PhytoLoss':        PhytoLinearLoss_Tani,
    'Grazing':          MatchedGrazing_Tani,
    'ZooLoss':          ZooLinearLoss_Tani,
})


# =============================================================================
# INPUT DICTIONARY
# =============================================================================
# Foreign label strings: 'N' (dissolved N), 'P' (phyto biomass),
# 'Z' (microzoo biomass), 'phyto_esd' / 'zoo_esd' (size grids).
input_vars = {
    'Nutrient': {
        'value_label': 'N',
        'value_init':  N_init,
    },
    'Phytoplankton': {
        'biomass_label':   'P',
        'biomass_init':    phyto_init,
        'phyto_esd_index': phyto_esd.tolist(),
        'phyto_esd_label': 'phyto_esd',
    },
    'Microzooplankton': {
        'biomass_label': 'Z',
        'biomass_init':  zoo_init,
        'zoo_esd_index': zoo_esd.tolist(),
        'zoo_esd_label': 'zoo_esd',
    },
    'Growth': {
        'resource': 'N',
        'consumer': 'P',
        'mu_max':   mu_max_arr,
        'halfsat':  k_s_arr,
    },
    'PhytoLoss': {
        'population': 'P',
        'nutrient':   'N',
        'rate':       lambda_val,
    },
    'Grazing': {
        'prey':      'P',
        'predator':  'Z',
        'nutrient':  'N',
        'g_max':     g_max_arr,
        'k_z':       k_z_arr,
        'gamma':     gamma_val,
    },
    'ZooLoss': {
        'population': 'Z',
        'nutrient':   'N',
        'rate':       delta_val,
    },
}


# =============================================================================
# MODEL SETUPS — three flavours for the three workflows
# =============================================================================
# Transient run with full output. Time span 5000 d matches the MS3
# convention and is well over the longest analytical relaxation
# timescale (set by Δ⁻¹ = 40 d for microzoo and Λ⁻¹ ≈ 670 d for
# phyto). Use for inspecting per-class time series and verifying
# convergence to the analytical steady-state predictions.
model_setup = xso.setup(
    solver='solve_ivp',
    model=model,
    time=np.arange(0, 5000, 1),
    input_vars=input_vars,
)

# Slim transient run — only the state variables and the key per-class
# fluxes needed for steady-state extraction and bin-metric computation.
# This is the setup that parscan workers should import for sweeps over
# N_T or any allometric exponent.
model_setup_slim = xso.setup(
    solver='solve_ivp',
    model=model,
    time=np.arange(0, 5000, 1),
    input_vars=input_vars,
    output_vars={
        'Nutrient__value',
        'Phytoplankton__biomass',
        'Microzooplankton__biomass',
        'Growth__uptake_value',
        'Grazing__loss_phyto_value',
        'Grazing__gain_zoo_value',
        'PhytoLoss__mortality_value',
        'ZooLoss__mortality_value',
    },
)

# Stability run — finds the steady state via fsolve, then numerically
# evaluates the Jacobian and reports eigenvalue-based stability
# classification. Use after the slim transient has produced a seed
# steady state; pair with `extract_steady_state_seed` (user-side, see
# `parscan_utils.py`) for warm-started fsolve.
model_setup_stability = xso.setup(
    solver='stability',
    model=model,
    time=[0, 1],
    input_vars=input_vars,
)
