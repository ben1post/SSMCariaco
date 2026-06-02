"""
Cariaco baseline NPZ model setups (Option A)
============================================
Iteration-1 IVP setup for the MS3 manuscript baseline:
Taniguchi 2014 Model 1 biology + Stock-style physical extensions.

Builds the model from cariaco_baseline_comps and creates an IVP setup for
size-spectrum diagnostic runs. F_N is the scan axis ('Supply__FN').

Iteration-1 parameter choices and their literature anchors:
- Allometries:  Taniguchi 2014 Table 1 verbatim (μ_max, k_s, I_max, K_sZ).
                MS3-defensible alternative (Stock/Hansen cluster, K_sZ
                uniform per Correction.md) flagged inline for swap.
- GGE Γ:        0.25 (Stock central / MS3-as-built; settled in Correction.md
                — scalar uniform per Hansen 1997 / Straile 1997)
- Pred:prey r:  10 (MS3/Stock/Banas convention, Survey §9; θ_opt=10 settled)
- Λ (phyto):    0.0015 d-1 (Taniguchi Table 1; tuned for Z:P, NOT for slope)
- Δ (zoo):      0.025 d-1 (Taniguchi Table 1; tuned for Z:P, NOT for slope)
- Grazing:      matched single-prey, Holling Type II (Taniguchi M1 verbatim)
- Supply:       Stock 2008 Eq. 7: F_N/d_e, no chemostat dilution
- Sinking:      w_sink/d_e = 0.1 d-1 (MS3-as-built: w_sink=5 m/d, d_e=50 m)
- No detritus, no fish kernel (iteration-1 baseline)
- Grid:         12 phyto (0.5-200 µm) + 12 zoo (5-2000 µm), log-spaced
                (MS3-as-built grid; Model Equations.md §1)

References:
- Taniguchi, Franks & Poulin 2014 MEPS 514:13-33 (Table 1 allometries)
- Cloern 2018 L&O 63:S392-S409 (Taniguchi-M1 adaptation precedent)
- Stock, Powell & Levin 2008 J. Mar. Syst. 74:134-152 (F_N/d_e, Eq. 7)
- Hansen, Bjørnsen & Hansen 1997 L&O 42:687-704 (allometric anchor)
- Correction.md (settled items: uniform K_sZ, GGE scalar, θ_opt=10)
- model context/Size Spectral Setup Survey.md §20.4 (layered build plan)
- model context/Taniguchi_Model1_Baseline.tex §7.3 (open-system bridge)
"""

import numpy as np
import xso

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from cariaco_baseline_comps import (
    Nutrient, PhytoSizeSpectrum, ZooSizeSpectrum,
    StockNutrientSupply,
    MonodGrowth_NP,
    PhytoLinearLoss_recycled, ZooLinearLoss_recycled,
    MatchedGrazing_TypeII, MatchedGrazing_TypeIII,
    PhytoSinking_export,
)

# Fish-variant reuses the MS3-as-built kernel component for one-way fish
# grazing. Both components in cariaco_ssm_comps.py, fully tested.
from cariaco_ssm_comps import FishGrazing_Kernel, ConstantFishForcing


# =============================================================================
# GRID  (MS3-as-built; Model Equations.md §1)
# =============================================================================
N_CLASSES        = 12
PHYTO_ESD_MIN    = 0.5      # µm
PHYTO_ESD_MAX    = 200.0    # µm
ZOO_PHYTO_RATIO  = 10.0     # r = 10 (Survey §9)

phyto_esd = np.logspace(np.log10(PHYTO_ESD_MIN),
                        np.log10(PHYTO_ESD_MAX), N_CLASSES)
zoo_esd   = ZOO_PHYTO_RATIO * phyto_esd


# =============================================================================
# ALLOMETRIES — TANIGUCHI 2014 TABLE 1 VERBATIM
# =============================================================================
# Alternative for MS3-defensible mesozoo-inclusive grid (Survey §6/§8):
#   Imax_arr = 26.0 * zoo_esd ** -0.48     # Stock/Hansen/Ward cluster
#   KsZ_arr  = np.full(N_CLASSES, 3.0)     # Hansen 1997, Correction.md settled

mu_max_arr = 1.36  * phyto_esd ** (-0.16)   # Taniguchi Eq. 7
ks_arr     = 0.33  * phyto_esd ** ( 0.48)   # Taniguchi Eq. 8
Imax_arr   = 33.96 * zoo_esd   ** (-0.66)   # Taniguchi Eq. 9
KsZ_arr    = 17.92 * zoo_esd   ** (-0.64)   # Taniguchi Eq. 10

LAMBDA_VAL = 0.0015         # phyto Λ [d-1], Taniguchi Table 1
DELTA_VAL  = 0.025          # zoo Δ   [d-1], Taniguchi Table 1
GAMMA_VAL  = 0.25           # Γ GGE   [-],  Stock / MS3-as-built

lambda_arr = np.full(N_CLASSES, LAMBDA_VAL)
delta_arr  = np.full(N_CLASSES, DELTA_VAL)


# =============================================================================
# STOCK PHYSICAL SETUP
# =============================================================================
FN_DEFAULT = 2.67           # mmol N m-2 d-1 ('all'-regime mean per cariaco_obs)
DE_DEFAULT = 50.0           # m, MS3-as-built box depth
W_SINK     = 5.0            # m/d, MS3-as-built sinking rate
W_OVER_DE  = W_SINK / DE_DEFAULT     # 0.1 d-1


# =============================================================================
# INITIAL CONDITIONS
# =============================================================================
N_INIT      = 1.0
P_INIT      = 1e-3
Z_INIT      = 1e-3
phyto_init  = np.full(N_CLASSES, P_INIT)
zoo_init    = np.full(N_CLASSES, Z_INIT)


# =============================================================================
# SOLVER CONFIG  (RK45 per 2026-05-22 finding: NPZ oscillatory ≠ stiff)
# =============================================================================
IVP_TIME_END   = 5000.0
ivp_time_array = np.arange(0.0, IVP_TIME_END, 1.0)

IVP_SOLVER_KWARGS = {
    'method': 'RK45',
    'atol': 1e-6, 'rtol': 1e-4,
    'instability_neg_threshold': -1e-3,
}


# =============================================================================
# MODEL — Option A: Taniguchi M1 biology + Stock supply + phyto sinking
# =============================================================================
# To swap functional response: replace 'Grazing': MatchedGrazing_TypeII
# with MatchedGrazing_TypeIII below. No other changes needed (same input
# slots, same output flux structure).
model_baseline = xso.create({
    'Nutrient':      Nutrient,
    'Phytoplankton': PhytoSizeSpectrum,
    'Zooplankton':   ZooSizeSpectrum,
    'Supply':        StockNutrientSupply,
    'Growth':        MonodGrowth_NP,
    'Grazing':       MatchedGrazing_TypeII,     # swap to MatchedGrazing_TypeIII
    'PhytoLoss':     PhytoLinearLoss_recycled,
    'ZooLoss':       ZooLinearLoss_recycled,
    'PhytoSinking':  PhytoSinking_export,
})


# =============================================================================
# SETUP — baseline IVP at FN_DEFAULT (Cariaco 'all'-regime mean)
# =============================================================================
# =============================================================================
# FISH VARIANT — simplified-fish power-law via existing FishGrazing_Kernel
# =============================================================================
# Reuses cariaco_ssm_comps.FishGrazing_Kernel (validated MS3-as-built component)
# with custom kernel arrays:
#   kernel_Z(s) = (s / s_max) ** e_F     — power-law on Z, peak=1 at largest
#   kernel_P(s) = 0                       — fish does not directly graze phyto
#
# Mechanism: fish suppresses large Z, which releases the matched prey (large P)
# from grazing. Per Taniguchi_Model1_Baseline.tex §10 item 5 (simplified-fish
# power-law) — preserves analytical tractability while concentrating top-down
# pressure on the mesozoo classes the Rykaczewski sardine kernel would target.
#
# Defaults chosen so fish_rate * fish_biomass * kernel_Z_max = 0.05 d-1 at the
# largest Z class — comparable to background Δ = 0.025 d-1, doubling per-capita
# loss on the top Z and creating a visible Micro-release signal.

FISH_E_F        = 1.5        # power-law exponent on Z kernel
FISH_RATE       = 0.05       # peak fish grazing rate per unit fish biomass [d-1]
FISH_BIOMASS    = 1.0        # constant fish forcing (dimensionless, MS3 convention)

kernel_Z_fish = (zoo_esd / zoo_esd.max()) ** FISH_E_F     # peak=1 at largest
kernel_P_fish = np.zeros(N_CLASSES)                        # no direct P grazing


model_baseline_fish = xso.create({
    'Nutrient':      Nutrient,
    'Phytoplankton': PhytoSizeSpectrum,
    'Zooplankton':   ZooSizeSpectrum,
    'Supply':        StockNutrientSupply,
    'Growth':        MonodGrowth_NP,
    'Grazing':       MatchedGrazing_TypeII,     # swap to MatchedGrazing_TypeIII
    'PhytoLoss':     PhytoLinearLoss_recycled,
    'ZooLoss':       ZooLinearLoss_recycled,
    'PhytoSinking':  PhytoSinking_export,
    'FishForcing':   ConstantFishForcing,        # NEW
    'FishGrazing':   FishGrazing_Kernel,         # NEW
})


# =============================================================================
# SETUP — baseline IVP at FN_DEFAULT (Cariaco 'all'-regime mean)
# =============================================================================
model_setup_baseline = xso.setup(
    solver='solve_ivp', model=model_baseline,
    time=ivp_time_array,
    input_vars={
        'Nutrient':      {'value_label': 'N', 'value_init': N_INIT},
        'Phytoplankton': {'biomass_label': 'P', 'biomass_init': phyto_init,
                          'phyto_esd_index': phyto_esd.tolist(),
                          'phyto_esd_label': 'phyto_esd'},
        'Zooplankton':   {'biomass_label': 'Z', 'biomass_init': zoo_init,
                          'zoo_esd_index': zoo_esd.tolist(),
                          'zoo_esd_label': 'zoo_esd'},
        'Supply':        {'var': 'N', 'FN': FN_DEFAULT, 'de': DE_DEFAULT},
        'Growth':        {'resource': 'N', 'consumer': 'P',
                          'mu_max': mu_max_arr, 'halfsat': ks_arr},
        'Grazing':       {'phyto': 'P', 'zoo': 'Z', 'nutrient': 'N',
                          'Imax': Imax_arr, 'KsZ': KsZ_arr,
                          'gamma': GAMMA_VAL},
        'PhytoLoss':     {'population': 'P', 'nutrient': 'N',
                          'rate': lambda_arr},
        'ZooLoss':       {'population': 'Z', 'nutrient': 'N',
                          'rate': delta_arr},
        'PhytoSinking':  {'population': 'P', 'w_over_de': W_OVER_DE},
    },
    solver_kwargs=IVP_SOLVER_KWARGS,
)


# =============================================================================
# SETUP — fish-variant IVP at FN_DEFAULT
# =============================================================================
model_setup_baseline_fish = xso.setup(
    solver='solve_ivp', model=model_baseline_fish,
    time=ivp_time_array,
    input_vars={
        'Nutrient':      {'value_label': 'N', 'value_init': N_INIT},
        'Phytoplankton': {'biomass_label': 'P', 'biomass_init': phyto_init,
                          'phyto_esd_index': phyto_esd.tolist(),
                          'phyto_esd_label': 'phyto_esd'},
        'Zooplankton':   {'biomass_label': 'Z', 'biomass_init': zoo_init,
                          'zoo_esd_index': zoo_esd.tolist(),
                          'zoo_esd_label': 'zoo_esd'},
        'Supply':        {'var': 'N', 'FN': FN_DEFAULT, 'de': DE_DEFAULT},
        'Growth':        {'resource': 'N', 'consumer': 'P',
                          'mu_max': mu_max_arr, 'halfsat': ks_arr},
        'Grazing':       {'phyto': 'P', 'zoo': 'Z', 'nutrient': 'N',
                          'Imax': Imax_arr, 'KsZ': KsZ_arr,
                          'gamma': GAMMA_VAL},
        'PhytoLoss':     {'population': 'P', 'nutrient': 'N',
                          'rate': lambda_arr},
        'ZooLoss':       {'population': 'Z', 'nutrient': 'N',
                          'rate': delta_arr},
        'PhytoSinking':  {'population': 'P', 'w_over_de': W_OVER_DE},
        'FishForcing':   {'forcing_label': 'F_forcing', 'value': FISH_BIOMASS},
        'FishGrazing':   {'phyto': 'P', 'zoo': 'Z',
                          'fish_forcing': 'F_forcing',
                          'kernel_P': kernel_P_fish,
                          'kernel_Z': kernel_Z_fish,
                          'rate': FISH_RATE},
    },
    solver_kwargs=IVP_SOLVER_KWARGS,
)


# =============================================================================
# TYPE III VARIANT — Option A baseline with Holling Type III matched grazing
# =============================================================================
# Identical to model_baseline / model_setup_baseline in every respect EXCEPT the
# grazing functional response (MatchedGrazing_TypeIII: low-prey refuge, G ∝ P²
# at low prey). Rohr 2022 (Survey §7): Type III far more stable than Type II
# (1.7% vs 37.5% of cases unstable). Same input_vars slots (Imax, KsZ, gamma) —
# no other change. Built for the Type II vs Type III stability assessment.
model_baseline_t3 = xso.create({
    'Nutrient':      Nutrient,
    'Phytoplankton': PhytoSizeSpectrum,
    'Zooplankton':   ZooSizeSpectrum,
    'Supply':        StockNutrientSupply,
    'Growth':        MonodGrowth_NP,
    'Grazing':       MatchedGrazing_TypeIII,
    'PhytoLoss':     PhytoLinearLoss_recycled,
    'ZooLoss':       ZooLinearLoss_recycled,
    'PhytoSinking':  PhytoSinking_export,
})

model_setup_baseline_t3 = xso.setup(
    solver='solve_ivp', model=model_baseline_t3,
    time=ivp_time_array,
    input_vars={
        'Nutrient':      {'value_label': 'N', 'value_init': N_INIT},
        'Phytoplankton': {'biomass_label': 'P', 'biomass_init': phyto_init,
                          'phyto_esd_index': phyto_esd.tolist(),
                          'phyto_esd_label': 'phyto_esd'},
        'Zooplankton':   {'biomass_label': 'Z', 'biomass_init': zoo_init,
                          'zoo_esd_index': zoo_esd.tolist(),
                          'zoo_esd_label': 'zoo_esd'},
        'Supply':        {'var': 'N', 'FN': FN_DEFAULT, 'de': DE_DEFAULT},
        'Growth':        {'resource': 'N', 'consumer': 'P',
                          'mu_max': mu_max_arr, 'halfsat': ks_arr},
        'Grazing':       {'phyto': 'P', 'zoo': 'Z', 'nutrient': 'N',
                          'Imax': Imax_arr, 'KsZ': KsZ_arr,
                          'gamma': GAMMA_VAL},
        'PhytoLoss':     {'population': 'P', 'nutrient': 'N',
                          'rate': lambda_arr},
        'ZooLoss':       {'population': 'Z', 'nutrient': 'N',
                          'rate': delta_arr},
        'PhytoSinking':  {'population': 'P', 'w_over_de': W_OVER_DE},
    },
    solver_kwargs=IVP_SOLVER_KWARGS,
)
