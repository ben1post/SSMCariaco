"""
Cariaco baseline NPZ model setups — MODERNISED 2026-06-06
=========================================================
Settled baseline construct (see `MS3 Project Background.md` 2026-06-06
"BASELINE CONSTRUCT SETTLED" block and `Baseline Construction Options.md`):
distributed Holling Type III grazing + distributed quadratic Z closure +
Stock F_N/d_e supply + phyto sinking, no detritus, no fish. Allometries moved
off Taniguchi-microzoo onto the coherent Banas (2011) / Ward (2012) /
Mattern (2026) lineage.

NOTE: the previous Taniguchi-verbatim setups file is kept under a separate
name as a historical reference. This file is now the working baseline.

Builds the model from cariaco_baseline_comps and creates IVP, slim-IVP, and
steady-state/stability setups for size-spectrum diagnostic runs and parameter
scans. F_N is the scan axis ('Inflow__FN'); the regime forcing dict returned by
cariaco_obs.load_cariaco_targets ({'Inflow__FN', 'Inflow__de'}) drops straight
in as fixed_overrides / input_vars_override.

Settled parameter choices and their literature anchors (Survey §§3-11):
- μ_max:        2.6·s^-0.45  (Tang 1995 / Banas 2011, monotonic)
- K_s:          0.144·s^0.81 (Ward 2012 / Marañón 2013 × Aksnes-Egge 1991)
- I_max:        26·s^-0.48   (Stock/Hansen/Ward/Mattern broad-span e_g)
- K_sZ:         UNIFORM ≈0.5 (Hansen/Rohr/Mattern/Dutkiewicz; Type III REQUIRES
                uniform K_sZ — Taniguchi allometric K_sZ collapses Type III)
- m_P (phyto):  0.1·μ_max    (Banas 2011, size-dependent)
- Z closure:    quadratic m_Z=0.1 (Banas 2011), recycled to N
- GGE Γ:        0.25 (Stock; range 0.25-0.33)
- Pred:prey r:  10 (θ_opt=10); kernel σ_log=0.25
- Grazing:      distributed Type III (DistributedGrazing_TypeIII); matched
                models retained as diagnostics (now on modern allometries)
- Supply:       Stock 2008 Eq. 7: F_N/d_e ('Inflow') + phyto sinking w=5 m/d
- No detritus, no fish kernel (baseline; fish = first layered addition)
- Grid:         ≥40 phyto (0.2-200 µm) + zoo (10× phyto), log-spaced;
                N_CLASSES via env-var MS3_N_CLASSES (default 40)
- Obs-fit knobs (Baseline Construction Options.md §3): K_sZ, e_g, Γ

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
    # Temperature (Cloern 2018 Q10 on growth + grazing) — folded into R0 (2026-06-06).
    ConstantTemperatureForcing, MonodGrowth_T, DistributedGrazing_TypeIII_T,
    # Diatom growth boost (piecewise mu_max branch) — 2026-06-08.
    MonodGrowth_Diatom_T,
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
# ALLOMETRIES — Banas (2011) / Ward (2012) / Mattern (2026) coherent lineage
# (modernised 2026-06-06; Taniguchi-verbatim values kept in the renamed
#  reference file. Sources: Survey §§3-6,8; Baseline Construction Options.md §2)
# LEGACY-TANIGUCHI (reference only):
# =============================================================================
# Alternative for MS3-defensible mesozoo-inclusive grid (Survey §6/§8):
#   Imax_arr = 26.0 * zoo_esd ** -0.48     # Stock/Hansen/Ward cluster
#   KsZ_arr  = np.full(N_CLASSES, 3.0)     # Hansen 1997, Correction.md settled

#   mu_max_arr = 1.36  * phyto_esd ** (-0.16)   # Taniguchi Eq. 7
#   ks_arr     = 0.33  * phyto_esd ** ( 0.48)   # Taniguchi Eq. 8
#   Imax_arr   = 33.96 * zoo_esd   ** (-0.66)   # Taniguchi Eq. 9
#   KsZ_arr    = 17.92 * zoo_esd   ** (-0.64)   # Taniguchi Eq. 10
#   lambda_arr = np.full(N_CLASSES, 0.0015)     # Taniguchi const Λ

mu_max_arr = 2.6   * phyto_esd ** (-0.45)   # Tang 1995 / Banas 2011 (was 1.36·s^-0.16)
ks_arr     = 0.144 * phyto_esd ** ( 0.81)   # Ward 2012 / Marañón×Aksnes-Egge (was 0.33·s^0.48)
Imax_arr   = 26.0  * zoo_esd   ** (-0.48)   # Stock/Hansen/Ward/Mattern e_g=-0.48 (was 33.96·z^-0.66)

# K_sZ: uniform and low. Type III REQUIRES uniform K_sZ — the Taniguchi
# allometric form collapses Type III (small-prey predators sit permanently in
# the S²-refuge; see Baseline Construction Options.md §5). Value in the
# literature window 0.15-0.5 mmol N m-3 (≈ Dutkiewicz 2015a); calibrate per obs.
KSZ_UNIFORM = 0.3   # temperature-inclusive obs-fit (2026-06-06): a single uniform
                    # K_sZ≈0.3 fits BOTH regimes' mean cell size within ~15% at the
                    # observed F_N (upwelling 4.7/obs 5.5, relaxed 2.3/obs 2.0 µm),
                    # stable (CV≤0.008). In the Rohr/Dutkiewicz window. (No-temp fit
                    # was 0.25; temperature enlarges cells → slightly higher K_sZ.)
KsZ_arr     = np.full(N_CLASSES, KSZ_UNIFORM)   # (was 17.92·z^-0.64 allometric)

DELTA_VAL  = 0.025          # zoo Δ [d-1] linear closure — matched reference models only
GAMMA_VAL  = 0.25           # Γ GGE [-],  Stock / MS3-as-built (range 0.25-0.33)

lambda_arr = 0.1 * mu_max_arr        # m_P = 0.1·μ_max (Banas 2011; size-dependent; was const 0.0015)
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
# THE SETTLED BASELINE CONSTRUCT (2026-06-06): distributed (kernel) grazing
# (matched / herb / omni via the phiPZ mode) × {Type II, Type III}, with the
# Banas-style distributed quadratic Z closure. Stock supply + sinking, modern
# allometries (μ/k_s/I_max above), and UNIFORM low K_sZ (KsZ_arr; Type III
# requires it). All grazing params are per-class arrays so the choice is always
# explicit and cannot silently become a scalar. Default baseline = herb or omni
# Type III; matched mode is the cross-check against the matched reference models.
#
# Exposed at module scope as model_setup_dist_<mode>_<resp>_{slim,stability}
# for run_xso_parscan / run_xso_stabilityscan (model-by-name contract).

THETA_OPT = 10.0    # predator:prey ESD ratio (matches ZOO_PHYTO_RATIO)
SIGMA_LOG     = 0.25   # ORIGINAL kernel width (log10 ESD, 2σ² convention). The
                       # matched/herb/omni baseline kernels keep historical behavior.
SIGMA_MATTERN = 0.15   # Mattern (2026) Eq.5 σ in the /σ² convention (= std 0.106),
                       # for the Mattern-faithful omnivory construct only.
M_Z_VAL   = 0.3     # quadratic closure coeff [(mmol N m-3)^-1 d-1]. ≈ Banas (2011)
                    # Eq.10 analytical estimate for THIS allometry family (~0.26;
                    # Survey §20.2) — an order below his abstract paper value 1.0.
                    # Stabilises the high-F_N oscillation the modern (faster Tang/Banas
                    # μ) allometries produce, while preserving large-cell extent
                    # (m_Z=0.1 oscillates CV→0.7; m_Z=1.0 over-damps → mcs<4µm).
                    # Was 0.1; raised 2026-06-06. Fit knob: range ~0.3-0.5.

# Feeding-preference matrices — only phiPZ differs across the three modes.
phiPZ_matched = compute_grazing_kernel(phyto_esd, zoo_esd, 'matched',
                                       THETA_OPT, SIGMA_LOG)
phiPZ_herb    = compute_grazing_kernel(phyto_esd, zoo_esd, 'herb',
                                       THETA_OPT, SIGMA_LOG)
phiPZ_omni    = compute_grazing_kernel(phyto_esd, zoo_esd, 'omni',
                                       THETA_OPT, SIGMA_LOG)
# Mattern-faithful omnivory kernel (σ=0.15, /σ² convention; = std 0.106). Built
# separately so the original-convention kernels above are untouched.
phiPZ_omni_mattern = compute_grazing_kernel(phyto_esd, zoo_esd, 'omni',
                                            THETA_OPT, SIGMA_MATTERN, convention='mattern')

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
    no silent per-variant parameter drift). Imax_arr is the modern allometry
    (26·z^-0.48); KsZ_arr is UNIFORM (0.5) — required for Type III."""
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


# =============================================================================
# TEMPERATURE-AWARE R0 — distributed Type III + Q10 growth/grazing (Cloern 2018)
# =============================================================================
# Folded into R0 2026-06-06: growth ×= 1.62^((T-20)/10), grazing ×= 2.48^((T-20)/10).
# Temperature is a per-regime forcing (box-mean Temp_C from cariaco_obs); override
# Temperature__value per regime (upwelling cooler, relaxed warmer). The Q10 effect
# in this model pushes the spectrum UP (more grazing crops small cells), so the
# obs-fit K_sZ must be RE-FIT with temperature ON (likely higher than the no-temp
# 0.25 — KSZ_UNIFORM=0.5 is the working default pending the re-fit).
# Temperature: Cloern (2018, L&O 63:S392) simple Q10 on growth AND grazing, in
# the Q10^((T-20)/10) form (Eqs 4, 6). Cloern is the closest 0D analogue (a
# Taniguchi-M1 + temperature estuarine model) and is the right complexity for a
# 0D box. Same principle as Dutkiewicz/Mattern (temperature on both, grazing more
# T-sensitive than growth) but without their Arrhenius + per-type thermal-norm
# machinery, which would over-specify a single-N box. (Earlier draft used the
# Eppley 1.066^T growth value — that is the EMPOWER/Anderson choice, not these
# three references — now reverted to Cloern.)
Q10_GROW  = 1.62          # Cloern 2018 Eq.4 — phyto growth Q10
Q10_GRAZE = 2.48          # Cloern 2018 Eq.6 — grazing Q10 (> growth; the size-
                          # selective term that actually moves the spectrum).
                          # Dutkiewicz/Mattern use ≈2.8. Set =1.0 for growth-only T.
T_REF     = 20.0          # °C, Cloern reference temperature
T_DEFAULT = 24.0    # °C, placeholder (overridden per regime via Temperature__value)

model_dist_t3_T = xso.create({
    'Nutrient':      Nutrient,
    'Phytoplankton': PhytoSizeSpectrum,
    'Zooplankton':   ZooSizeSpectrum,
    'Inflow':        StockNutrientSupply,
    'Temperature':   ConstantTemperatureForcing,
    'Growth':        MonodGrowth_T,
    'Grazing':       DistributedGrazing_TypeIII_T,
    'GrazingRouter': DistributedGrazingRouter,
    'PhytoLoss':     PhytoLinearLoss_recycled,
    'ZooLoss':       ZooQuadraticLoss_recycled,
    'PhytoSinking':  PhytoSinking_export,
})


def make_dist_T_input_vars(phiPZ):
    """Temperature-aware input_vars: reuse the base distributed dict and add the
    Temperature forcing + Q10/T_ref to Growth and Grazing. Only phiPZ varies
    across kernel modes (same enforced-identical-params principle)."""
    iv = make_dist_input_vars(phiPZ)
    iv['Temperature'] = {'forcing_label': 'temperature', 'value': T_DEFAULT}
    iv['Growth']  = {**iv['Growth'],  'temperature': 'temperature',
                     'q10': Q10_GROW,  't_ref': T_REF}
    iv['Grazing'] = {**iv['Grazing'], 'temperature': 'temperature',
                     'q10': Q10_GRAZE, 't_ref': T_REF}
    return iv


def _distT_slim(phiPZ):
    return xso.setup(solver='solve_ivp', model=model_dist_t3_T, time=ivp_time_array,
                     input_vars=make_dist_T_input_vars(phiPZ),
                     output_vars=SLIM_OUTPUT_VARS, solver_kwargs=IVP_SOLVER_KWARGS)

def _distT_stab(phiPZ):
    return xso.setup(solver='stability', model=model_dist_t3_T, time=STAB_TIME,
                     input_vars=make_dist_T_input_vars(phiPZ))

# R0 (temperature-aware) herb / omni setups
model_setup_dist_herb_t3_T_slim      = _distT_slim(phiPZ_herb)
model_setup_dist_herb_t3_T_stability = _distT_stab(phiPZ_herb)
model_setup_dist_omni_t3_T_slim      = _distT_slim(phiPZ_omni)
model_setup_dist_omni_t3_T_stability = _distT_stab(phiPZ_omni)

# Mattern-faithful omnivory R0 (σ=0.15 /σ² kernel = std 0.106) — the construct for
# the omnivory work. Original omni setups above keep the σ=0.25 (2σ²) kernel.
model_setup_dist_omni_t3_T_mattern_slim      = _distT_slim(phiPZ_omni_mattern)
model_setup_dist_omni_t3_T_mattern_stability = _distT_stab(phiPZ_omni_mattern)


# =============================================================================
# LINEAR + QUADRATIC zoo closure (Mattern SI S4.2) — adds ZooLinearLoss alongside
# ZooLoss(quadratic). Mattern uses both, uniform (non-allometric); linear mortality
# is their explicit large-Z biomass control lever (SI Fig S8). M_Z_LIN uniform here
# (Taniguchi Δ=0.025 / Cloern 0.06 d-1); per-class array, so it can be made
# size-dependent later (e.g. reduce on large Z, Mattern-style).
# =============================================================================
M_Z_LIN_DEFAULT = 0.05    # linear zoo mortality Δ [d-1], uniform

model_dist_t3_T_zlin = xso.create({
    'Nutrient':      Nutrient,
    'Phytoplankton': PhytoSizeSpectrum,
    'Zooplankton':   ZooSizeSpectrum,
    'Inflow':        StockNutrientSupply,
    'Temperature':   ConstantTemperatureForcing,
    'Growth':        MonodGrowth_T,
    'Grazing':       DistributedGrazing_TypeIII_T,
    'GrazingRouter': DistributedGrazingRouter,
    'PhytoLoss':     PhytoLinearLoss_recycled,
    'ZooLoss':       ZooQuadraticLoss_recycled,
    'ZooLossLin':    ZooLinearLoss_recycled,
    'PhytoSinking':  PhytoSinking_export,
})

def make_dist_T_zlin_input_vars(phiPZ, m_lin=M_Z_LIN_DEFAULT):
    iv = make_dist_T_input_vars(phiPZ)
    iv['ZooLossLin'] = {'population': 'Z', 'nutrient': 'N',
                        'rate': np.full(N_CLASSES, m_lin)}   # per-class (dims='zoo')
    return iv

def _distT_zlin_slim(phiPZ, m_lin=M_Z_LIN_DEFAULT):
    return xso.setup(solver='solve_ivp', model=model_dist_t3_T_zlin, time=ivp_time_array,
                     input_vars=make_dist_T_zlin_input_vars(phiPZ, m_lin),
                     output_vars=SLIM_OUTPUT_VARS, solver_kwargs=IVP_SOLVER_KWARGS)

def _distT_zlin_stab(phiPZ, m_lin=M_Z_LIN_DEFAULT):
    return xso.setup(solver='stability', model=model_dist_t3_T_zlin, time=STAB_TIME,
                     input_vars=make_dist_T_zlin_input_vars(phiPZ, m_lin))

# omni-Mattern + linear&quadratic zoo closure
model_setup_dist_omni_t3_T_zlin_slim      = _distT_zlin_slim(phiPZ_omni_mattern)
model_setup_dist_omni_t3_T_zlin_stability = _distT_zlin_stab(phiPZ_omni_mattern)

# Diagnostic variant: loosened instability floor (-1e-2) so small transient
# negatives during spin-up don't abort the run — for the m_lin solve_ivp scan.
# Genuine divergence still NaN-terminates (positive ceiling / non-finite).
model_setup_dist_omni_t3_T_zlin_slim_loose = xso.setup(
    solver='solve_ivp', model=model_dist_t3_T_zlin, time=ivp_time_array,
    input_vars=make_dist_T_zlin_input_vars(phiPZ_omni_mattern),
    output_vars=SLIM_OUTPUT_VARS,
    solver_kwargs={**IVP_SOLVER_KWARGS, 'instability_neg_threshold': -1e-2})


# =============================================================================
# DIATOM GROWTH BOOST — piecewise ('diatom') mu_max branch on the T-aware R0
# =============================================================================
# 2026-06-08: fish grazing is structurally inert in herbivory R0 because the
# 2-level web can't sustain large Z (no large-P stock to feed them). The lever
# is a diatom GROWTH boost, not grazing-defence: a flatter mu_max branch above
# a size threshold builds a standing large-P stock -> seeds large Z (cascade
# prerequisite). Swaps Growth -> MonodGrowth_Diatom_T; everything else is R0.
# diatom_exp / diatom_thresh are scalar params -> parscan-able. K_s and m_P stay
# on the base allometry (affinity penalty preserved; m_P-coupling caveat noted
# in the component docstring). diatom_exp == BASE_EXP reproduces R0 exactly.
MU0_VAL               = 2.6      # mu_max prefactor (Tang 1995 / Banas 2011)
BASE_EXP              = -0.45    # small-cell mu_max exponent (R0)
DIATOM_EXP_DEFAULT    = -0.45    # default == BASE_EXP -> recovers R0
DIATOM_THRESH_DEFAULT = 20.0     # crossover ESD [µm] (Nano/Micro boundary)

model_dist_t3_T_diatom = xso.create({
    'Nutrient':      Nutrient,
    'Phytoplankton': PhytoSizeSpectrum,
    'Zooplankton':   ZooSizeSpectrum,
    'Inflow':        StockNutrientSupply,
    'Temperature':   ConstantTemperatureForcing,
    'Growth':        MonodGrowth_Diatom_T,
    'Grazing':       DistributedGrazing_TypeIII_T,
    'GrazingRouter': DistributedGrazingRouter,
    'PhytoLoss':     PhytoLinearLoss_recycled,
    'ZooLoss':       ZooQuadraticLoss_recycled,
    'PhytoSinking':  PhytoSinking_export,
})


def make_dist_T_diatom_input_vars(phiPZ, diatom_exp=DIATOM_EXP_DEFAULT,
                                  diatom_thresh=DIATOM_THRESH_DEFAULT):
    """T-aware input_vars with the piecewise-diatom Growth block. Fully REPLACES
    the Growth entry (the new component takes esd/mu0/base_exp/diatom_* instead
    of the precomputed mu_max array); all other blocks are the R0-T defaults."""
    iv = make_dist_T_input_vars(phiPZ)
    iv['Growth'] = {'resource': 'N', 'consumer': 'P', 'temperature': 'temperature',
                    'esd': phyto_esd, 'mu0': MU0_VAL, 'base_exp': BASE_EXP,
                    'diatom_exp': diatom_exp, 'diatom_thresh': diatom_thresh,
                    'halfsat': ks_arr, 'q10': Q10_GROW, 't_ref': T_REF}
    return iv


def _distT_diatom_slim(phiPZ, diatom_exp=DIATOM_EXP_DEFAULT,
                       diatom_thresh=DIATOM_THRESH_DEFAULT):
    return xso.setup(solver='solve_ivp', model=model_dist_t3_T_diatom, time=ivp_time_array,
                     input_vars=make_dist_T_diatom_input_vars(phiPZ, diatom_exp, diatom_thresh),
                     output_vars=SLIM_OUTPUT_VARS, solver_kwargs=IVP_SOLVER_KWARGS)

def _distT_diatom_stab(phiPZ, diatom_exp=DIATOM_EXP_DEFAULT,
                       diatom_thresh=DIATOM_THRESH_DEFAULT):
    return xso.setup(solver='stability', model=model_dist_t3_T_diatom, time=STAB_TIME,
                     input_vars=make_dist_T_diatom_input_vars(phiPZ, diatom_exp, diatom_thresh))

# Diatom-boost herb setups (defaults reproduce R0; scan overrides Growth__diatom_*)
model_setup_dist_herb_t3_T_diatom_slim      = _distT_diatom_slim(phiPZ_herb)
model_setup_dist_herb_t3_T_diatom_stability = _distT_diatom_stab(phiPZ_herb)


# =============================================================================
# FISH (R0 + Rykaczewski sardine kernel) — first top-down layered addition
# =============================================================================
# Full Rykaczewski (2019) two-sigmoid sardine clearance-rate curve, evaluated on
# BOTH grids and jointly peak-normalised (= cariaco_ssm_setup, MS3-as-built). Zoo
# ESD passed directly (no prosome ×2.5), no filter-feeding clamp — exactly the
# original calls. Fish biomass F prescribed constant; one-way export (catch).
# Mechanism: kernel peaks ~1230 µm → crops large Z → releases the large P those Z
# grazed → candidate Nano→Micro carve via the cascade.

def clearance_rate_sardine_vdl(prey_length_um, filter_feeding=False):
    """Sardine size-specific clearance rate, Rykaczewski (2019) Eq. 3 (two-sigmoid)."""
    x = np.asarray(prey_length_um, dtype=float)
    def _f(xv):
        e1 = np.exp(0.0198 * (xv - 15.0))
        term1 = (9.03 * e1) / (12.03 + 0.75 * e1)
        e2 = np.exp(0.00843 * (xv - 800.0))
        term2 = (9.96 * e2) / (30.8 + 0.323 * e2)
        return term1 + term2
    F_S = _f(x)
    if filter_feeding:
        F_S = np.where(x > 1230.0, _f(np.array(1230.0)), F_S)
    return F_S

def compute_fish_kernel_vdl_joint(phyto_esd, zoo_esd):
    """Rykaczewski clearance on P and Z grids, JOINTLY peak-normalised (peak=1
    on whichever grid holds the curve maximum — typically the large-zoo end)."""
    F_P = clearance_rate_sardine_vdl(phyto_esd)
    F_Z = clearance_rate_sardine_vdl(zoo_esd)
    F_max = max(F_P.max(), F_Z.max())
    return F_P / F_max, F_Z / F_max

FISH_BIOMASS = 1.0      # prescribed constant fish biomass (MS3-as-built)
FISH_RATE    = 0.005    # peak fish grazing rate per unit fish biomass [d-1] (MS3-as-built)
kernel_P_fish, kernel_Z_fish = compute_fish_kernel_vdl_joint(phyto_esd, zoo_esd)

model_dist_t3_T_fish = xso.create({
    'Nutrient':      Nutrient,
    'Phytoplankton': PhytoSizeSpectrum,
    'Zooplankton':   ZooSizeSpectrum,
    'Inflow':        StockNutrientSupply,
    'Temperature':   ConstantTemperatureForcing,
    'Growth':        MonodGrowth_T,
    'Grazing':       DistributedGrazing_TypeIII_T,
    'GrazingRouter': DistributedGrazingRouter,
    'PhytoLoss':     PhytoLinearLoss_recycled,
    'ZooLoss':       ZooQuadraticLoss_recycled,
    'PhytoSinking':  PhytoSinking_export,
    'FishForcing':   ConstantFishForcing,
    'FishGrazing':   FishGrazing_Kernel,
})


def make_dist_T_fish_input_vars(phiPZ):
    """Temperature-aware R0 input_vars + the Rykaczewski fish kernel."""
    iv = make_dist_T_input_vars(phiPZ)
    iv['FishForcing'] = {'forcing_label': 'F_forcing', 'value': FISH_BIOMASS}
    iv['FishGrazing'] = {'phyto': 'P', 'zoo': 'Z', 'fish_forcing': 'F_forcing',
                         'kernel_P': kernel_P_fish, 'kernel_Z': kernel_Z_fish,
                         'rate': FISH_RATE}
    return iv


def _distTf_slim(phiPZ):
    return xso.setup(solver='solve_ivp', model=model_dist_t3_T_fish, time=ivp_time_array,
                     input_vars=make_dist_T_fish_input_vars(phiPZ),
                     output_vars=SLIM_OUTPUT_VARS, solver_kwargs=IVP_SOLVER_KWARGS)

def _distTf_stab(phiPZ):
    return xso.setup(solver='stability', model=model_dist_t3_T_fish, time=STAB_TIME,
                     input_vars=make_dist_T_fish_input_vars(phiPZ))

model_setup_dist_herb_t3_T_fish_slim      = _distTf_slim(phiPZ_herb)
model_setup_dist_herb_t3_T_fish_stability = _distTf_stab(phiPZ_herb)
model_setup_dist_omni_t3_T_fish_slim      = _distTf_slim(phiPZ_omni)
model_setup_dist_omni_t3_T_fish_stability = _distTf_stab(phiPZ_omni)


# =============================================================================
# DIATOM GROWTH BOOST + FISH — the construct for the F_N x fish crosswise
# =============================================================================
# 2026-06-08: the diatom boost (MonodGrowth_Diatom_T) seeds a stable large-Z
# population at bloom F_N (z95 ~985 µm, fZ>200 ~0.3), closing the size-scale gap
# that made fish inert in R0. This variant = diatom-boost Growth + the same
# Rykaczewski kernel, so the F_N x fish crosswise (the core MS3 experiment) runs
# on a stable, large-Z-bearing state. diatom_exp / diatom_thresh and the fish
# rate are all scalar params -> parscan-able.

model_dist_t3_T_diatom_fish = xso.create({
    'Nutrient':      Nutrient,
    'Phytoplankton': PhytoSizeSpectrum,
    'Zooplankton':   ZooSizeSpectrum,
    'Inflow':        StockNutrientSupply,
    'Temperature':   ConstantTemperatureForcing,
    'Growth':        MonodGrowth_Diatom_T,
    'Grazing':       DistributedGrazing_TypeIII_T,
    'GrazingRouter': DistributedGrazingRouter,
    'PhytoLoss':     PhytoLinearLoss_recycled,
    'ZooLoss':       ZooQuadraticLoss_recycled,
    'PhytoSinking':  PhytoSinking_export,
    'FishForcing':   ConstantFishForcing,
    'FishGrazing':   FishGrazing_Kernel,
})


def make_dist_T_diatom_fish_input_vars(phiPZ, diatom_exp=DIATOM_EXP_DEFAULT,
                                       diatom_thresh=DIATOM_THRESH_DEFAULT):
    """Diatom-boost Growth block (from make_dist_T_diatom_input_vars) + the
    Rykaczewski fish kernel (from make_dist_T_fish_input_vars)."""
    iv = make_dist_T_diatom_input_vars(phiPZ, diatom_exp, diatom_thresh)
    iv['FishForcing'] = {'forcing_label': 'F_forcing', 'value': FISH_BIOMASS}
    iv['FishGrazing'] = {'phyto': 'P', 'zoo': 'Z', 'fish_forcing': 'F_forcing',
                         'kernel_P': kernel_P_fish, 'kernel_Z': kernel_Z_fish,
                         'rate': FISH_RATE}
    return iv


def _distTdf_slim(phiPZ, diatom_exp=DIATOM_EXP_DEFAULT, diatom_thresh=DIATOM_THRESH_DEFAULT):
    return xso.setup(solver='solve_ivp', model=model_dist_t3_T_diatom_fish, time=ivp_time_array,
                     input_vars=make_dist_T_diatom_fish_input_vars(phiPZ, diatom_exp, diatom_thresh),
                     output_vars=SLIM_OUTPUT_VARS, solver_kwargs=IVP_SOLVER_KWARGS)

def _distTdf_stab(phiPZ, diatom_exp=DIATOM_EXP_DEFAULT, diatom_thresh=DIATOM_THRESH_DEFAULT):
    return xso.setup(solver='stability', model=model_dist_t3_T_diatom_fish, time=STAB_TIME,
                     input_vars=make_dist_T_diatom_fish_input_vars(phiPZ, diatom_exp, diatom_thresh))

model_setup_dist_herb_t3_T_diatom_fish_slim      = _distTdf_slim(phiPZ_herb)
model_setup_dist_herb_t3_T_diatom_fish_stability = _distTdf_stab(phiPZ_herb)
