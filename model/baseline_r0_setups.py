"""
Cariaco N-P-Z-D baseline model — SETUPS (baseline_r0, 2026-06-14).

Clean setup module for the manuscript baseline. Imports the refactored
components from baseline_r0_comps (scalar K_sZ; scalar sigma_log and
theta_opt with phiPZ computed via setup_func; scalar uniform phyto
mortality rate).

Models (4):
  - model_baseline                 : bulk Z closure + uniform phyto mortality
  - model_baseline_perclassZ       : per-class Z closure + uniform phyto mortality
  - model_baseline_banas           : bulk Z closure + Banas (coeff·μ_max) phyto mortality
  - model_baseline_perclassZ_banas : per-class Z closure + Banas phyto mortality

Setups per model (3):
  - setup_<name>             : full output, default solver
  - setup_<name>_slim        : slim output (6 vars), default solver
  - setup_<name>_stability   : stability solver

NO loosened instability guards baked into any setup. Pass solver_kwargs at
run/scan time where required (Benny, 2026-06-14).

Defaults:
  Growth (Taniguchi 2014 Table 1, ESD-native):
    μ_max = 1.36·s^(-0.16) d⁻¹,  K_s = 0.33·s^(0.48) mmol N m⁻³
  Phyto mortality:
    uniform m_P = 0.02 d⁻¹ (Ward-like)
    Banas coeff = 0.1 (10% of μ_max; Banas 2011)
  Grazing (Dutkiewicz 2020 + Mattern 2026):
    I_max piecewise (9.8 d⁻¹ for zoo ≤30 µm; 30.9·V^(-0.16) above)
    K_sZ = 0.23 mmol N m⁻³ (scalar, uniform — Dutkiewicz k_p=1.5 mmolC ÷6.625)
    σ = 0.15 (Mattern ÷σ² convention)
    θ_opt = 10
    GGE = 0.25 (Stock 2008)
  Zoo closure:
    bulk quad m_Z = 0.1 (Banas-style; Z_j·ΣZ)
    per-class quad m_Z = 1.4 (Dutkiewicz 2015a Table 6: 22.4 d⁻¹·m³(mmolP)⁻¹
                              ÷16 Redfield N:P)
    linear m_Zlin = 0.05 (Dutkiewicz range ~0.067)
  Detritus: k_remin = 0.1 d⁻¹, w_sink = 5 m d⁻¹
  Temperature (Cloern 2018): Q10_grow=1.62, Q10_graze=2.48, T_ref=20 °C
  Fish: r_F = 0 (no fish; top-down lever scanned in MS3 crosswise)
"""

import os
import numpy as np
import xso
from xso.parscans import avg_tail  # re-export so run_xso_parscan can find it by name

from baseline_r0_comps import (
    Nutrient, PhytoSizeSpectrum, ZooSizeSpectrum, Detritus,
    StockNutrientSupply, ConstantTemperatureForcing,
    MonodGrowth_T,
    DistributedGrazing_TypeIII_T, DistributedGrazing_TypeIII_T_Herb,
    DistributedGrazingRouter_route,
    PhytoMortality_route, BanasPhytoMortality_route,
    ZooLinearMortality_route, ZooQuadraticMortality_route,
    ZooQuadraticMortality_perclass_route,
    DetritusRemineralization, DetritusSinking, FishGrazing_Kernel_rate,
    compute_fish_kernel_vdl_joint,
)

# =============================================================================
# Size grid (phyto 0.2–200 µm, zoo = 10·phyto)
# =============================================================================
N_CLASSES       = int(os.environ.get('MS3_BASELINE_N', 40))   # reported runs ≥40
PHYTO_ESD_MIN   = 0.2      # µm (Sieburth Pico floor)
PHYTO_ESD_MAX   = 200.0    # µm
ZOO_PHYTO_RATIO = 10.0     # zoo_esd = 10·phyto_esd (θ_opt)
phyto_esd = np.logspace(np.log10(PHYTO_ESD_MIN), np.log10(PHYTO_ESD_MAX), N_CLASSES)
zoo_esd   = ZOO_PHYTO_RATIO * phyto_esd

# =============================================================================
# Allometries — default = Taniguchi 2014 (growth) + Dutkiewicz 2020 (grazing)
# =============================================================================
# Phyto growth — Taniguchi 2014 Table 1, ESD-native:
mu_max_arr = 1.36 * phyto_esd ** (-0.16)        # d⁻¹
ks_arr     = 0.33 * phyto_esd ** ( 0.48)        # mmol N m⁻³

# Zoo grazing I_max — Dutkiewicz 2020 Suppl. Table S2, V in µm³:
# zoo ≤30 µm: 9.8 d⁻¹ constant (smallest 4 grazers); zoo >30 µm: 30.9·V^-0.16
zoo_vol  = (np.pi / 6.0) * zoo_esd ** 3
Imax_arr = np.where(zoo_esd <= 30.0, 9.8, 30.9 * zoo_vol ** (-0.16))

# =============================================================================
# Scalar parameters
# =============================================================================
# --- grazing ---
K_SZ            = 0.23      # grazing half-sat [mmol N m-3] (Dutkiewicz k_p=1.5 mmolC ÷6.625)
SIGMA           = 0.15      # kernel width σ (Mattern 2026 ÷σ² convention)
THETA_OPT       = 10.0      # predator:prey ESD ratio
GGE_VAL         = 0.25      # gross growth efficiency (Stock 2008)

# --- phyto mortality ---
M_P             = 0.02      # uniform phyto mortality [d-1] (Ward-like)
BANAS_MORT_COEFF = 0.1      # Banas coeff (m_P = coeff·μ_max; Banas 2011, 10% of μ_max)

# --- zoo closure ---
M_Z_BULK        = 0.1       # bulk quad m_Z·Z_j·ΣZ [(mmolN m-3)^-1 d-1] (Banas-style)
M_Z_PERCLASS    = 1.4       # per-class quad m_Z·Z_j² [(mmolN m-3)^-1 d-1]
                            # Dutkiewicz 2015a Table 6: 22.4 d⁻¹·m³(mmolP)⁻¹ ÷ 16 (Redfield N:P)
M_ZLIN          = 0.05      # linear zoo mortality/excretion [d-1] (Dutkiewicz range ~0.067)

# --- detritus ---
K_REMIN         = 0.1       # remineralisation rate [d-1] (warm tropical)
W_SINK          = 5.0       # m/d (detritus sinking velocity)

# --- temperature (Cloern 2018) — growth + grazing only ---
Q10_GROW        = 1.62
Q10_GRAZE       = 2.48
T_REF           = 20.0      # °C (allometry reference)
T_DEFAULT       = 24.0      # °C placeholder (override per regime)

# --- supply (Stock 2008) — override per regime ---
FN_DEFAULT      = 2.67      # mmol N m-2 d-1 ('all'-regime mean)
DE_DEFAULT      = 50.0      # m
# d_e is BROADCAST (single source of truth): override Inflow__de per regime;
# DetritusSink foreign-references it so one d_e drives both F_N/d_e and w_sink/d_e.

# --- fish (top-down lever) — Rykaczewski kernel, scalar rate ---
FISH_RATE       = 0.0       # default = no-fish baseline; scan up for the crosswise
kernel_P_fish, kernel_Z_fish = compute_fish_kernel_vdl_joint(phyto_esd, zoo_esd)

# --- loss-fate routing (frac_D, frac_export; remainder -> N) ---
GRAZE_FRAC_D            = 0.75   # unassimilated grazing -> D (Fasham); rest -> N
GRAZE_FRAC_EXPORT       = 0.0
PHYTO_MORT_FRAC_D       = 0.9    # phyto mortality -> D; rest -> N (Fasham-style)
PHYTO_MORT_FRAC_EXPORT  = 0.0
ZOO_LIN_FRAC_D          = 1.0    # linear Z mortality -> 100% D (Benny, 2026-06-09)
ZOO_LIN_FRAC_EXPORT     = 0.0
ZOO_QUAD_FRAC_D         = 0.5    # quadratic closure: 50% D, 50% export (Stock-style)
ZOO_QUAD_FRAC_EXPORT    = 0.5

# =============================================================================
# Initial conditions and solver / time
# =============================================================================
N_INIT  = 1.0
P_INIT  = 1e-3
Z_INIT  = 1e-3
D_INIT  = 1e-3
phyto_init = np.full(N_CLASSES, P_INIT)
zoo_init   = np.full(N_CLASSES, Z_INIT)

# RK45 + relaxed atol only (NOT LSODA). NO loosened instability guard baked in;
# pass solver_kwargs at run/scan time where needed (e.g. for bloom-F_N scans).
IVP_SOLVER_KWARGS = {'method': 'RK45', 'atol': 1e-6, 'rtol': 1e-4}
IVP_TIME_END   = 5000.0
ivp_time_array = np.arange(0.0, IVP_TIME_END, 1.0)
STAB_TIME      = [0.0, 1.0]

# Slim output: state vars + the two flux diagnostics we read from every scan
SLIM_OUTPUT_VARS = {
    'Nutrient__value',
    'Phytoplankton__biomass',
    'Zooplankton__biomass',
    'Detritus__value',
    'Growth__uptake_value',
    'DetritusSink__sinking_value',
}

# =============================================================================
# Models — 4 variants
# =============================================================================

model_baseline = xso.create({
    'Nutrient':         Nutrient,
    'Phytoplankton':    PhytoSizeSpectrum,
    'Zooplankton':      ZooSizeSpectrum,
    'Detritus':         Detritus,
    'Inflow':           StockNutrientSupply,
    'Temperature':      ConstantTemperatureForcing,
    'Growth':           MonodGrowth_T,
    'Grazing':          DistributedGrazing_TypeIII_T,
    'GrazingRouter':    DistributedGrazingRouter_route,
    'PhytoMortality':   PhytoMortality_route,
    'ZooLinMortality':  ZooLinearMortality_route,
    'ZooQuadMortality': ZooQuadraticMortality_route,
    'DetritusRemin':    DetritusRemineralization,
    'DetritusSink':     DetritusSinking,
    'FishGrazing':      FishGrazing_Kernel_rate,
})

model_baseline_perclassZ = xso.create({
    'Nutrient':         Nutrient,
    'Phytoplankton':    PhytoSizeSpectrum,
    'Zooplankton':      ZooSizeSpectrum,
    'Detritus':         Detritus,
    'Inflow':           StockNutrientSupply,
    'Temperature':      ConstantTemperatureForcing,
    'Growth':           MonodGrowth_T,
    'Grazing':          DistributedGrazing_TypeIII_T,
    'GrazingRouter':    DistributedGrazingRouter_route,
    'PhytoMortality':   PhytoMortality_route,
    'ZooLinMortality':  ZooLinearMortality_route,
    'ZooQuadMortality': ZooQuadraticMortality_perclass_route,   # <- swap vs model_baseline
    'DetritusRemin':    DetritusRemineralization,
    'DetritusSink':     DetritusSinking,
    'FishGrazing':      FishGrazing_Kernel_rate,
})

model_baseline_banas = xso.create({
    'Nutrient':         Nutrient,
    'Phytoplankton':    PhytoSizeSpectrum,
    'Zooplankton':      ZooSizeSpectrum,
    'Detritus':         Detritus,
    'Inflow':           StockNutrientSupply,
    'Temperature':      ConstantTemperatureForcing,
    'Growth':           MonodGrowth_T,
    'Grazing':          DistributedGrazing_TypeIII_T,
    'GrazingRouter':    DistributedGrazingRouter_route,
    'PhytoMortality':   BanasPhytoMortality_route,              # <- swap vs model_baseline
    'ZooLinMortality':  ZooLinearMortality_route,
    'ZooQuadMortality': ZooQuadraticMortality_route,
    'DetritusRemin':    DetritusRemineralization,
    'DetritusSink':     DetritusSinking,
    'FishGrazing':      FishGrazing_Kernel_rate,
})

model_baseline_perclassZ_banas = xso.create({
    'Nutrient':         Nutrient,
    'Phytoplankton':    PhytoSizeSpectrum,
    'Zooplankton':      ZooSizeSpectrum,
    'Detritus':         Detritus,
    'Inflow':           StockNutrientSupply,
    'Temperature':      ConstantTemperatureForcing,
    'Growth':           MonodGrowth_T,
    'Grazing':          DistributedGrazing_TypeIII_T,
    'GrazingRouter':    DistributedGrazingRouter_route,
    'PhytoMortality':   BanasPhytoMortality_route,              # <- two swaps vs model_baseline
    'ZooLinMortality':  ZooLinearMortality_route,
    'ZooQuadMortality': ZooQuadraticMortality_perclass_route,
    'DetritusRemin':    DetritusRemineralization,
    'DetritusSink':     DetritusSinking,
    'FishGrazing':      FishGrazing_Kernel_rate,
})


# =============================================================================
# Input-vars builders
# =============================================================================
def make_baseline_input_vars(fish_rate=FISH_RATE, FN=FN_DEFAULT, de=DE_DEFAULT,
                             T=T_DEFAULT, mu_max=mu_max_arr, halfsat=ks_arr,
                             mP=M_P, m_Z=M_Z_BULK):
    """Input-vars for the baseline model with UNIFORM PhytoMortality_route.

    Override fish_rate / FN / de / T per regime or scan. `de` is broadcast
    (single source of truth) — overriding it drives both the supply F_N/d_e
    and the detritus sinking w_sink/d_e. For the per-class closure variant
    pass m_Z=M_Z_PERCLASS at the call site (or override ZooQuadMortality__rate
    via fixed_overrides at run time).
    """
    return {
        'Nutrient':         {'value_label': 'N', 'value_init': N_INIT},
        'Phytoplankton':    {'biomass_label': 'P', 'biomass_init': phyto_init,
                             'phyto_esd_index': phyto_esd.tolist(),
                             'phyto_esd_label': 'phyto_esd'},
        'Zooplankton':      {'biomass_label': 'Z', 'biomass_init': zoo_init,
                             'zoo_esd_index': zoo_esd.tolist(),
                             'zoo_esd_label': 'zoo_esd'},
        'Detritus':         {'value_label': 'D', 'value_init': D_INIT},
        'Inflow':           {'var': 'N', 'FN': FN, 'de': de, 'de_label': 'de'},
        'Temperature':      {'forcing_label': 'temperature', 'value': T},
        'Growth':           {'resource': 'N', 'consumer': 'P', 'temperature': 'temperature',
                             'mu_max_label': 'mu_max', 'mu_max': mu_max, 'halfsat': halfsat,
                             'q10': Q10_GROW, 't_ref': T_REF},
        'Grazing':          {'resource': 'P', 'consumer': 'Z', 'temperature': 'temperature',
                             'phyto_esd': 'phyto_esd', 'zoo_esd': 'zoo_esd',
                             'theta_opt': THETA_OPT, 'sigma_log': SIGMA,
                             'Imax': Imax_arr, 'KsZ': K_SZ,
                             'q10': Q10_GRAZE, 't_ref': T_REF},
        'GrazingRouter':    {'grazed_phyto': 'P', 'grazed_zoo': 'Z',
                             'assimilated_consumer': 'Z', 'egested_detritus': 'D',
                             'excreted_nutrient': 'N', 'gge': GGE_VAL,
                             'frac_D': GRAZE_FRAC_D, 'frac_export': GRAZE_FRAC_EXPORT},
        'PhytoMortality':   {'population': 'P', 'detritus': 'D', 'nutrient': 'N',
                             'rate': mP,
                             'frac_D': PHYTO_MORT_FRAC_D,
                             'frac_export': PHYTO_MORT_FRAC_EXPORT},
        'ZooLinMortality':  {'population': 'Z', 'detritus': 'D', 'nutrient': 'N',
                             'rate': M_ZLIN, 'frac_D': ZOO_LIN_FRAC_D,
                             'frac_export': ZOO_LIN_FRAC_EXPORT},
        'ZooQuadMortality': {'population': 'Z', 'detritus': 'D', 'nutrient': 'N',
                             'rate': m_Z, 'frac_D': ZOO_QUAD_FRAC_D,
                             'frac_export': ZOO_QUAD_FRAC_EXPORT},
        'DetritusRemin':    {'detritus': 'D', 'nutrient': 'N', 'k_remin': K_REMIN},
        'DetritusSink':     {'detritus': 'D', 'w_sink': W_SINK, 'de': 'de'},
        'FishGrazing':      {'phyto': 'P', 'zoo': 'Z',
                             'kernel_P': kernel_P_fish, 'kernel_Z': kernel_Z_fish,
                             'rate': fish_rate},
    }


def make_baseline_banas_input_vars(fish_rate=FISH_RATE, FN=FN_DEFAULT, de=DE_DEFAULT,
                                    T=T_DEFAULT, mu_max=mu_max_arr, halfsat=ks_arr,
                                    coeff=BANAS_MORT_COEFF, m_Z=M_Z_BULK):
    """Input-vars for the Banas-mortality variant: replaces the PhytoMortality
    slot with the Banas form (foreign μ_max + scalar coeff). All other slots
    identical to make_baseline_input_vars. Pass m_Z=M_Z_PERCLASS for the
    per-class closure variant."""
    iv = make_baseline_input_vars(fish_rate=fish_rate, FN=FN, de=de, T=T,
                                   mu_max=mu_max, halfsat=halfsat, m_Z=m_Z)
    iv['PhytoMortality'] = {'population': 'P', 'detritus': 'D', 'nutrient': 'N',
                            'mu_max': 'mu_max', 'coeff': coeff,
                            'frac_D': PHYTO_MORT_FRAC_D,
                            'frac_export': PHYTO_MORT_FRAC_EXPORT}
    return iv


# =============================================================================
# Setups — 4 models × 3 setups each = 12 total
# =============================================================================

# ---- model_baseline ---------------------------------------------------------
setup_baseline = xso.setup(
    solver='solve_ivp', model=model_baseline, time=ivp_time_array,
    input_vars=make_baseline_input_vars(),
    solver_kwargs=IVP_SOLVER_KWARGS)

setup_baseline_slim = xso.setup(
    solver='solve_ivp', model=model_baseline, time=ivp_time_array,
    input_vars=make_baseline_input_vars(), output_vars=SLIM_OUTPUT_VARS,
    solver_kwargs=IVP_SOLVER_KWARGS)

setup_baseline_stability = xso.setup(
    solver='stability', model=model_baseline, time=STAB_TIME,
    input_vars=make_baseline_input_vars())

# ---- model_baseline_perclassZ ----------------------------------------------
setup_baseline_perclassZ = xso.setup(
    solver='solve_ivp', model=model_baseline_perclassZ, time=ivp_time_array,
    input_vars=make_baseline_input_vars(m_Z=M_Z_PERCLASS),
    solver_kwargs=IVP_SOLVER_KWARGS)

setup_baseline_perclassZ_slim = xso.setup(
    solver='solve_ivp', model=model_baseline_perclassZ, time=ivp_time_array,
    input_vars=make_baseline_input_vars(m_Z=M_Z_PERCLASS),
    output_vars=SLIM_OUTPUT_VARS, solver_kwargs=IVP_SOLVER_KWARGS)

setup_baseline_perclassZ_stability = xso.setup(
    solver='stability', model=model_baseline_perclassZ, time=STAB_TIME,
    input_vars=make_baseline_input_vars(m_Z=M_Z_PERCLASS))

# ---- model_baseline_banas --------------------------------------------------
setup_baseline_banas = xso.setup(
    solver='solve_ivp', model=model_baseline_banas, time=ivp_time_array,
    input_vars=make_baseline_banas_input_vars(),
    solver_kwargs=IVP_SOLVER_KWARGS)

setup_baseline_banas_slim = xso.setup(
    solver='solve_ivp', model=model_baseline_banas, time=ivp_time_array,
    input_vars=make_baseline_banas_input_vars(), output_vars=SLIM_OUTPUT_VARS,
    solver_kwargs=IVP_SOLVER_KWARGS)

setup_baseline_banas_stability = xso.setup(
    solver='stability', model=model_baseline_banas, time=STAB_TIME,
    input_vars=make_baseline_banas_input_vars())

# ---- model_baseline_perclassZ_banas ----------------------------------------
setup_baseline_perclassZ_banas = xso.setup(
    solver='solve_ivp', model=model_baseline_perclassZ_banas, time=ivp_time_array,
    input_vars=make_baseline_banas_input_vars(m_Z=M_Z_PERCLASS),
    solver_kwargs=IVP_SOLVER_KWARGS)

setup_baseline_perclassZ_banas_slim = xso.setup(
    solver='solve_ivp', model=model_baseline_perclassZ_banas, time=ivp_time_array,
    input_vars=make_baseline_banas_input_vars(m_Z=M_Z_PERCLASS),
    output_vars=SLIM_OUTPUT_VARS, solver_kwargs=IVP_SOLVER_KWARGS)

setup_baseline_perclassZ_banas_stability = xso.setup(
    solver='stability', model=model_baseline_perclassZ_banas, time=STAB_TIME,
    input_vars=make_baseline_banas_input_vars(m_Z=M_Z_PERCLASS))

# =============================================================================
# Herbivory variant (added 2026-06-14 for cross-regime tension diagnostic)
# =============================================================================
# Zoo eat phyto only (no zoo-on-zoo). Same scalar K_sZ + setup_func phiPZ +
# Q10 structure as the main model; only the kernel mode differs.
# Use to test whether the cross-regime tension under sustained F_N stems
# from omnivory (Taniguchi Model 3 non-monotonic mcs(F_N) phenomenon).
model_baseline_herb = xso.create({
    'Nutrient':         Nutrient,
    'Phytoplankton':    PhytoSizeSpectrum,
    'Zooplankton':      ZooSizeSpectrum,
    'Detritus':         Detritus,
    'Inflow':           StockNutrientSupply,
    'Temperature':      ConstantTemperatureForcing,
    'Growth':           MonodGrowth_T,
    'Grazing':          DistributedGrazing_TypeIII_T_Herb,    # <- swap vs model_baseline
    'GrazingRouter':    DistributedGrazingRouter_route,
    'PhytoMortality':   PhytoMortality_route,
    'ZooLinMortality':  ZooLinearMortality_route,
    'ZooQuadMortality': ZooQuadraticMortality_route,
    'DetritusRemin':    DetritusRemineralization,
    'DetritusSink':     DetritusSinking,
    'FishGrazing':      FishGrazing_Kernel_rate,
})

setup_baseline_herb = xso.setup(
    solver='solve_ivp', model=model_baseline_herb, time=ivp_time_array,
    input_vars=make_baseline_input_vars(),
    solver_kwargs=IVP_SOLVER_KWARGS)

setup_baseline_herb_slim = xso.setup(
    solver='solve_ivp', model=model_baseline_herb, time=ivp_time_array,
    input_vars=make_baseline_input_vars(), output_vars=SLIM_OUTPUT_VARS,
    solver_kwargs=IVP_SOLVER_KWARGS)

setup_baseline_herb_stability = xso.setup(
    solver='stability', model=model_baseline_herb, time=STAB_TIME,
    input_vars=make_baseline_input_vars())
