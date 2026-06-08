"""
Cariaco baseline NPZ model setups (Option A)
============================================
Iteration-1 setups for the MS3 manuscript baseline:
Taniguchi 2014 Model 1 biology + Stock-style physical extensions.

Builds the model from cariaco_baseline_comps and creates IVP, slim-IVP, and
steady-state/stability setups for size-spectrum diagnostic runs and parameter
scans. F_N is the scan axis ('Inflow__FN'); the regime forcing dict returned by
cariaco_obs.load_cariaco_targets ({'Inflow__FN', 'Inflow__de'}) drops straight
in as fixed_overrides / input_vars_override.

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
- Supply:       Stock 2008 Eq. 7: F_N/d_e (component label 'Inflow')
- Sinking:      w_sink = 5 m/d; d_e is broadcast from Inflow and foreign-
                referenced by PhytoSinking, so the export rate w_sink/d_e
                follows the one regime depth without a w_over_de param.
- No detritus, no fish kernel (iteration-1 baseline)
- Grid:         12 phyto (0.5-200 µm) + 12 zoo (5-2000 µm), log-spaced
                (MS3-as-built grid; Model Equations.md §1)

Setups exposed at module scope (parscan contract — xso.parscans imports these
by name): per model variant a full-output IVP setup (single diagnostic runs),
a slim-output IVP setup (parameter scans), and a stability setup (fsolve +
eigenvalues, time=[0,1]).

References:
- Taniguchi, Franks & Poulin 2014 MEPS 514:13-33 (Table 1 allometries)
- Cloern 2018 L&O 63:S392-S409 (Taniguchi-M1 adaptation precedent)
- Stock, Powell & Levin 2008 J. Mar. Syst. 74:134-152 (F_N/d_e, Eq. 7)
- Hansen, Bjørnsen & Hansen 1997 L&O 42:687-704 (allometric anchor)
- Correction.md (settled items: uniform K_sZ, GGE scalar, θ_opt=10)
- model context/Size Spectral Setup Survey.md §20.4 (layered build plan)
- model context/Taniguchi_Model1_Baseline.tex §7.3 (open-system bridge)
"""

import os
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
    # Distributed (kernel) grazing family + quadratic closure (added 2026-06).
    # Per-class Imax AND KsZ — no scalar regression possible.
    compute_grazing_kernel,
    DistributedGrazing_TypeII, DistributedGrazing_TypeIII,
    DistributedGrazingRouter,
    ZooQuadraticLoss_recycled,
)

# Fish-variant reuses the MS3-as-built kernel component for one-way fish
# grazing. Both components in cariaco_ssm_comps.py, fully tested.
from cariaco_ssm_comps import FishGrazing_Kernel, ConstantFishForcing

# Required for parscans that pass postprocess_name='avg_tail': run_xso_parscan
# resolves the postprocess by ATTRIBUTE LOOKUP on this model module (the
# worker does getattr(module, postprocess_name)), so avg_tail must live in
# this module's namespace or the pre-flight check fails with
# "Failed to find postprocess callable 'avg_tail'".
from xso.parscans import avg_tail


# =============================================================================
# GRID  (MS3-as-built; Model Equations.md §1)
# =============================================================================
N_CLASSES        = int(os.environ.get('MS3_N_CLASSES', 40))
                           # Default 40 (≥40 reporting standard; 12 = diagnostic-only).
                           # Override per-run WITHOUT editing this file by setting
                           # os.environ['MS3_N_CLASSES'] = '80' in the notebook BEFORE
                           # the parscan imports the module (workers re-import on fork
                           # and pick it up) — lets resolution scans run via parscan.
                           # At 40 the σ_log=0.25 kernel spans ~6.5 grid cells so
                           # herb/omni are genuinely distributed (at 12 the kernel ≈
                           # one cell wide → herb/omni collapse to matched).
PHYTO_ESD_MIN    = 0.2      # µm — canonical Sieburth Pico floor (0.5 was outdated)
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
# d_e is a single broadcast parameter on the Inflow component; PhytoSinking
# foreign-references it, so supply (F_N/d_e) and sinking (w_sink/d_e) both
# follow from one regime depth. No W_OVER_DE constant.
FN_DEFAULT = 2.67           # mmol N m-2 d-1 ('all'-regime mean per cariaco_obs)
DE_DEFAULT = 50.0           # m, MS3-as-built box depth
W_SINK     = 5.0            # m/d, MS3-as-built sinking velocity


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

# Stability solver runs fsolve + eigenvalues; XSO convention is a length-2
# time array (initial state at [0], steady state at [-1]).
STAB_TIME = [0.0, 1.0]

# Slim output for parameter scans — state variables + the two flux targets
# (PP, export) + the forcing parameters (Inflow__de needed by parscan_utils
# to convert volumetric export to areal). Keeps parscan worker output small.
SLIM_OUTPUT_VARS = {
    'Nutrient__value',
    'Phytoplankton__biomass',
    'Zooplankton__biomass',
    'Growth__uptake_value',
    'PhytoSinking__sinking_value',
    'Inflow__FN',
    'Inflow__de',
}


# =============================================================================
# FISH VARIANT — simplified power-law kernel via existing FishGrazing_Kernel
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


# =============================================================================
# MODELS — Option A: Taniguchi M1 biology + Stock supply + phyto sinking
# =============================================================================
# Type II and Type III differ only in the Grazing component class; every other
# component (and every input slot) is identical, so they share one input_vars
# dict below. To add fish, use model_baseline_fish.
model_baseline = xso.create({
    'Nutrient':      Nutrient,
    'Phytoplankton': PhytoSizeSpectrum,
    'Zooplankton':   ZooSizeSpectrum,
    'Inflow':        StockNutrientSupply,
    'Growth':        MonodGrowth_NP,
    'Grazing':       MatchedGrazing_TypeII,
    'PhytoLoss':     PhytoLinearLoss_recycled,
    'ZooLoss':       ZooLinearLoss_recycled,
    'PhytoSinking':  PhytoSinking_export,
})

model_baseline_t3 = xso.create({
    'Nutrient':      Nutrient,
    'Phytoplankton': PhytoSizeSpectrum,
    'Zooplankton':   ZooSizeSpectrum,
    'Inflow':        StockNutrientSupply,
    'Growth':        MonodGrowth_NP,
    'Grazing':       MatchedGrazing_TypeIII,     # low-prey refuge (Rohr 2022)
    'PhytoLoss':     PhytoLinearLoss_recycled,
    'ZooLoss':       ZooLinearLoss_recycled,
    'PhytoSinking':  PhytoSinking_export,
})

model_baseline_fish = xso.create({
    'Nutrient':      Nutrient,
    'Phytoplankton': PhytoSizeSpectrum,
    'Zooplankton':   ZooSizeSpectrum,
    'Inflow':        StockNutrientSupply,
    'Growth':        MonodGrowth_NP,
    'Grazing':       MatchedGrazing_TypeII,
    'PhytoLoss':     PhytoLinearLoss_recycled,
    'ZooLoss':       ZooLinearLoss_recycled,
    'PhytoSinking':  PhytoSinking_export,
    'FishForcing':   ConstantFishForcing,
    'FishGrazing':   FishGrazing_Kernel,
})


# =============================================================================
# SHARED INPUT_VARS
# =============================================================================
# One dict serves Type II and Type III (same component labels and slots; only
# the Grazing class differs between the two models). Spelled out in full here;
# the variant setups below reuse it rather than re-listing every slot.
#
# Forcing wiring:
#   Inflow.de is broadcast under the label 'de' (slots de_label + de);
#   PhytoSinking.de foreign-references that label ('de'). Overriding the value
#   slot 'Inflow__de' (e.g. via the regime forcing dict) drives both supply
#   and sinking — nothing else to touch.
baseline_input_vars = {
    'Nutrient':      {'value_label': 'N', 'value_init': N_INIT},
    'Phytoplankton': {'biomass_label': 'P', 'biomass_init': phyto_init,
                      'phyto_esd_index': phyto_esd.tolist(),
                      'phyto_esd_label': 'phyto_esd'},
    'Zooplankton':   {'biomass_label': 'Z', 'biomass_init': zoo_init,
                      'zoo_esd_index': zoo_esd.tolist(),
                      'zoo_esd_label': 'zoo_esd'},
    'Inflow':        {'var': 'N', 'FN': FN_DEFAULT,
                      'de': DE_DEFAULT, 'de_label': 'de'},
    'Growth':        {'resource': 'N', 'consumer': 'P',
                      'mu_max': mu_max_arr, 'halfsat': ks_arr},
    'Grazing':       {'phyto': 'P', 'zoo': 'Z', 'nutrient': 'N',
                      'Imax': Imax_arr, 'KsZ': KsZ_arr,
                      'gamma': GAMMA_VAL},
    'PhytoLoss':     {'population': 'P', 'nutrient': 'N',
                      'rate': lambda_arr},
    'ZooLoss':       {'population': 'Z', 'nutrient': 'N',
                      'rate': delta_arr},
    'PhytoSinking':  {'population': 'P', 'w_sink': W_SINK, 'de': 'de'},
}

# Fish variant = baseline slots + the two fish components.
fish_input_vars = {
    **baseline_input_vars,
    'FishForcing':   {'forcing_label': 'F_forcing', 'value': FISH_BIOMASS},
    'FishGrazing':   {'phyto': 'P', 'zoo': 'Z',
                      'fish_forcing': 'F_forcing',
                      'kernel_P': kernel_P_fish,
                      'kernel_Z': kernel_Z_fish,
                      'rate': FISH_RATE},
}


# =============================================================================
# SETUPS — Type II (baseline)
# =============================================================================
# Full-output IVP — single diagnostic runs (full per-class P/Z time series).
model_setup_baseline = xso.setup(
    solver='solve_ivp', model=model_baseline, time=ivp_time_array,
    input_vars=baseline_input_vars,
    solver_kwargs=IVP_SOLVER_KWARGS,
)

# Slim-output IVP — parameter scans (run_xso_parscan + avg_tail).
model_setup_baseline_slim = xso.setup(
    solver='solve_ivp', model=model_baseline, time=ivp_time_array,
    input_vars=baseline_input_vars, output_vars=SLIM_OUTPUT_VARS,
    solver_kwargs=IVP_SOLVER_KWARGS,
)

# Stability — fsolve steady state + Jacobian eigenvalues (run_xso_stabilityscan,
# seeded from an IVP tail-mean via parscan_utils.extract_steady_state_seed).
model_setup_baseline_stability = xso.setup(
    solver='stability', model=model_baseline, time=STAB_TIME,
    input_vars=baseline_input_vars,
)


# =============================================================================
# SETUPS — Type III (low-prey refuge variant)
# =============================================================================
model_setup_baseline_t3 = xso.setup(
    solver='solve_ivp', model=model_baseline_t3, time=ivp_time_array,
    input_vars=baseline_input_vars,
    solver_kwargs=IVP_SOLVER_KWARGS,
)

model_setup_baseline_t3_slim = xso.setup(
    solver='solve_ivp', model=model_baseline_t3, time=ivp_time_array,
    input_vars=baseline_input_vars, output_vars=SLIM_OUTPUT_VARS,
    solver_kwargs=IVP_SOLVER_KWARGS,
)

model_setup_baseline_t3_stability = xso.setup(
    solver='stability', model=model_baseline_t3, time=STAB_TIME,
    input_vars=baseline_input_vars,
)


# =============================================================================
# SETUP — fish variant (IVP only; slim/stability added when fish is in scope)
# =============================================================================
model_setup_baseline_fish = xso.setup(
    solver='solve_ivp', model=model_baseline_fish, time=ivp_time_array,
    input_vars=fish_input_vars,
    solver_kwargs=IVP_SOLVER_KWARGS,
)


# =============================================================================
# DISTRIBUTED-GRAZING VARIANTS — kernel grazing + quadratic closure
# =============================================================================
# Layered deviation beyond the matched baseline: distributed (kernel) grazing
# (matched / herb / omni via the phiPZ mode) × {Type II, Type III}, with the
# Banas-style distributed quadratic Z closure. Same Stock supply + sinking,
# same Taniguchi allometric μ/k_s, and — crucially — the SAME allometric
# Imax_arr (33.96·z^-0.66) and KsZ_arr (17.92·z^-0.64) as the matched baseline,
# passed as per-class arrays so the grazing allometry (the slope lever, Survey
# §18) is explicit and cannot silently become a scalar.
#
# Exposed at module scope as model_setup_dist_<mode>_<resp>_{slim,stability}
# for run_xso_parscan / run_xso_stabilityscan (model-by-name contract).

THETA_OPT = 10.0    # predator:prey ESD ratio (matches ZOO_PHYTO_RATIO)
SIGMA_LOG = 0.25    # kernel width, log10 ESD, 2σ² convention (Survey §9)
M_Z_VAL   = 0.1     # quadratic closure coeff [(mmol N m-3)^-1 d-1] (Banas/MS3)

# Feeding-preference matrices — only phiPZ differs across the three modes.
phiPZ_matched = compute_grazing_kernel(phyto_esd, zoo_esd, 'matched',
                                       THETA_OPT, SIGMA_LOG)
phiPZ_herb    = compute_grazing_kernel(phyto_esd, zoo_esd, 'herb',
                                       THETA_OPT, SIGMA_LOG)
phiPZ_omni    = compute_grazing_kernel(phyto_esd, zoo_esd, 'omni',
                                       THETA_OPT, SIGMA_LOG)

# Two model schemas (Type II vs Type III grazing matrix); the kernel mode is an
# input (phiPZ), not a structural change, so it is NOT baked into the model.
model_dist_t2 = xso.create({
    'Nutrient':      Nutrient,
    'Phytoplankton': PhytoSizeSpectrum,
    'Zooplankton':   ZooSizeSpectrum,
    'Inflow':        StockNutrientSupply,
    'Growth':        MonodGrowth_NP,
    'Grazing':       DistributedGrazing_TypeII,
    'GrazingRouter': DistributedGrazingRouter,
    'PhytoLoss':     PhytoLinearLoss_recycled,
    'ZooLoss':       ZooQuadraticLoss_recycled,
    'PhytoSinking':  PhytoSinking_export,
})

model_dist_t3 = xso.create({
    'Nutrient':      Nutrient,
    'Phytoplankton': PhytoSizeSpectrum,
    'Zooplankton':   ZooSizeSpectrum,
    'Inflow':        StockNutrientSupply,
    'Growth':        MonodGrowth_NP,
    'Grazing':       DistributedGrazing_TypeIII,
    'GrazingRouter': DistributedGrazingRouter,
    'PhytoLoss':     PhytoLinearLoss_recycled,
    'ZooLoss':       ZooQuadraticLoss_recycled,
    'PhytoSinking':  PhytoSinking_export,
})


def make_dist_input_vars(phiPZ):
    """Input-vars dict for a distributed-grazing run. Only phiPZ varies across
    the three kernel modes — every other parameter is shared, which ENFORCES
    identical params across the matched/herb/omni comparison (the whole point:
    no silent per-variant parameter drift). Imax_arr / KsZ_arr are the same
    Taniguchi allometric per-class arrays as the matched baseline."""
    return {
        'Nutrient':      {'value_label': 'N', 'value_init': N_INIT},
        'Phytoplankton': {'biomass_label': 'P', 'biomass_init': phyto_init,
                          'phyto_esd_index': phyto_esd.tolist(),
                          'phyto_esd_label': 'phyto_esd'},
        'Zooplankton':   {'biomass_label': 'Z', 'biomass_init': zoo_init,
                          'zoo_esd_index': zoo_esd.tolist(),
                          'zoo_esd_label': 'zoo_esd'},
        'Inflow':        {'var': 'N', 'FN': FN_DEFAULT,
                          'de': DE_DEFAULT, 'de_label': 'de'},
        'Growth':        {'resource': 'N', 'consumer': 'P',
                          'mu_max': mu_max_arr, 'halfsat': ks_arr},
        'Grazing':       {'resource': 'P', 'consumer': 'Z', 'phiPZ': phiPZ,
                          'Imax': Imax_arr, 'KsZ': KsZ_arr},
        'GrazingRouter': {'grazed_phyto': 'P', 'grazed_zoo': 'Z',
                          'assimilated_consumer': 'Z', 'excreted_nutrient': 'N',
                          'gamma': GAMMA_VAL},
        'PhytoLoss':     {'population': 'P', 'nutrient': 'N', 'rate': lambda_arr},
        'ZooLoss':       {'population': 'Z', 'nutrient': 'N', 'rate': M_Z_VAL},
        'PhytoSinking':  {'population': 'P', 'w_sink': W_SINK, 'de': 'de'},
    }


def _dist_slim(model, phiPZ):
    return xso.setup(solver='solve_ivp', model=model, time=ivp_time_array,
                     input_vars=make_dist_input_vars(phiPZ),
                     output_vars=SLIM_OUTPUT_VARS, solver_kwargs=IVP_SOLVER_KWARGS)

def _dist_stab(model, phiPZ):
    return xso.setup(solver='stability', model=model, time=STAB_TIME,
                     input_vars=make_dist_input_vars(phiPZ))

# Six constructs (mode × response), each with a slim IVP setup (parscan) and a
# stability setup (stabilityscan). Named explicitly so parscan can look them up.
model_setup_dist_matched_t2_slim      = _dist_slim(model_dist_t2, phiPZ_matched)
model_setup_dist_matched_t2_stability = _dist_stab(model_dist_t2, phiPZ_matched)
model_setup_dist_herb_t2_slim         = _dist_slim(model_dist_t2, phiPZ_herb)
model_setup_dist_herb_t2_stability    = _dist_stab(model_dist_t2, phiPZ_herb)
model_setup_dist_omni_t2_slim         = _dist_slim(model_dist_t2, phiPZ_omni)
model_setup_dist_omni_t2_stability    = _dist_stab(model_dist_t2, phiPZ_omni)

model_setup_dist_matched_t3_slim      = _dist_slim(model_dist_t3, phiPZ_matched)
model_setup_dist_matched_t3_stability = _dist_stab(model_dist_t3, phiPZ_matched)
model_setup_dist_herb_t3_slim         = _dist_slim(model_dist_t3, phiPZ_herb)
model_setup_dist_herb_t3_stability    = _dist_stab(model_dist_t3, phiPZ_herb)
model_setup_dist_omni_t3_slim         = _dist_slim(model_dist_t3, phiPZ_omni)
model_setup_dist_omni_t3_stability    = _dist_stab(model_dist_t3, phiPZ_omni)
