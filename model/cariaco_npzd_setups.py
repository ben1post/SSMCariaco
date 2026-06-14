"""
Cariaco N-P-Z-D size-spectrum model — SETUPS (final baseline, 2026-06-09).

Clean, single-model setup for the corrected previous-MS3 baseline. Replaces
the old cariaco_baseline_setups.py (archive that as *_old). Full spec:
model context/MS3_Final_Baseline_Model.tex.

Construct: N-P-Z-D + detritus, distributed Holling Type III OMNIVORY, distributed
quadratic + linear Z closure, linear P mortality, Stock F_N/d_e supply per regime,
detritus remineralisation + sinking, Cloern temperature (growth + grazing only),
Rykaczewski sardine grazing as a scalar rate (default 0 = no-fish baseline).

Allometries — single source by functional response:
  GROWTH  = Banas 2011 Table 2:  μ_max = 2.6·s^-0.45,  K_s = 0.1·s^1.0,  m_P = 0.1·μ_max
  GRAZING = Dutkiewicz 2020:      g_max = 9.8 (zoo ≤30µm) / 30.9·V^-0.16 (>30µm),
                                  K_sZ = 0.23 (uniform), σ = 0.15 (Mattern kernel)
Scalars: GGE = 0.25 (Stock), m_Z = 0.1 (quad), m_Zlin = 0.05 (linear), θ_opt = 10.

Every loss on P/Z is freely routable N/D/export via (frac_D, frac_export); the
basic setup routes linear-Z mortality 100% to D (Benny, 2026-06-09). Vary the
fractions / m_Zlin / m_Z to test mortality-form and routing combinations.
"""

import os
import numpy as np
import xso
from xso.parscans import avg_tail  # re-export so run_xso_parscan can find it by name

from cariaco_npzd_comps import (
    Nutrient, PhytoSizeSpectrum, ZooSizeSpectrum, Detritus,
    StockNutrientSupply, ConstantTemperatureForcing,
    MonodGrowth_T, DistributedGrazing_TypeIII_T, DistributedGrazingRouter_route,
    PhytoMortality_route, BanasPhytoMortality_route,
    ZooLinearMortality_route, ZooQuadraticMortality_route,
    ZooQuadraticMortality_perclass_route,
    DetritusRemineralization, DetritusSinking, FishGrazing_Kernel_rate,
    compute_grazing_kernel, compute_fish_kernel_vdl_joint,
)

# ---- grid -------------------------------------------------------------------
N_CLASSES       = int(os.environ.get('MS3_NPZD_N', 40))   # reported runs ≥40
PHYTO_ESD_MIN   = 0.2      # µm (Sieburth Pico floor)
PHYTO_ESD_MAX   = 200.0    # µm
ZOO_PHYTO_RATIO = 10.0     # zoo_esd = 10·phyto_esd (θ_opt)
phyto_esd = np.logspace(np.log10(PHYTO_ESD_MIN), np.log10(PHYTO_ESD_MAX), N_CLASSES)
zoo_esd   = ZOO_PHYTO_RATIO * phyto_esd

# ---- allometries: two-source split by functional response -------------------
# GROWTH = Banas 2011 (Monod-native); GRAZING = Dutkiewicz 2020 (Type-III-native).
mu_max_arr = 2.6 * phyto_esd ** (-0.45)        # Banas 2011 Table 2 (←Tang 1995)
ks_arr     = 0.1 * phyto_esd ** ( 1.0)         # Banas 2011 Table 2 (Eppley 1969)
mP_arr     = 0.1 * mu_max_arr                  # m_P = 0.1·μ_max (Banas 2011)
# Grazing g_max (Dutkiewicz 2020 Table S2, V in µm³): zoo ≤30µm = 9.8 d⁻¹ const
# (4 smallest grazers); zoo >30µm = 30.9·V^-0.16. Step at 30µm is by design.
zoo_vol  = (np.pi / 6.0) * zoo_esd ** 3
Imax_arr = np.where(zoo_esd <= 30.0, 9.8, 30.9 * zoo_vol ** (-0.16))

# ---- ALTERNATIVE growth: Taniguchi 2014 Table 1 (added 2026-06-10) -----------
# Banas growth (steep μ + surface-area K_s extrapolated to the 0.2µm Pico floor)
# collapses the spectrum to Pico; Taniguchi's own-synthesis μ/K_s (ESD-native,
# fit over 0.8µm–1.9mm) give a coexisting, F_N-responsive Pico↔Micro spectrum.
mu_max_tani = 1.36 * phyto_esd ** (-0.16)      # d⁻¹        (Taniguchi 2014 Table 1)
ks_tani     = 0.33 * phyto_esd ** ( 0.48)      # mmol N m⁻³ (Taniguchi 2014 Table 1)
mP_tani     = 0.1 * mu_max_tani                # Banas rule on Taniguchi μ (consistent default)

# ---- scalar params ----------------------------------------------------------
GGE_VAL    = 0.25      # gross growth efficiency (Stock 2008)
K_SZ       = 0.23      # grazing half-sat [mmol N m-3], uniform (Dutkiewicz k_p ÷6.625)
SIGMA      = 0.15      # kernel width σ (Mattern 2026, ÷σ² convention)
THETA_OPT  = 10.0      # predator:prey ESD ratio
M_Z        = 0.1       # quadratic closure coeff [(mmol N m-3)^-1 d-1] (Banas 2011)
M_ZLIN     = 0.05      # linear zoo mortality/excretion [d-1] (Dutkiewicz range ~0.067)
KsZ_arr    = np.full(N_CLASSES, K_SZ)

# loss-fate routing fractions (frac_D, frac_export; remainder -> N) ------------
GRAZE_FRAC_D        = 0.75   # unassimilated grazing -> D (Fasham); rest -> N
GRAZE_FRAC_EXPORT   = 0.0
PHYTO_MORT_FRAC_D   = 0.9    # phyto mortality -> D; rest -> N (Fasham-style)
PHYTO_MORT_FRAC_EXPORT = 0.0
BANAS_MORT_COEFF    = 0.1    # Banas non-grazing mortality = 10% of μ_max (m_P=coeff·μ_max)
ZOO_LIN_FRAC_D      = 1.0    # linear Z mortality -> 100% D (basic setup, Benny)
ZOO_LIN_FRAC_EXPORT = 0.0
ZOO_QUAD_FRAC_D     = 0.5    # quadratic closure: 50% D, 50% export (Stock-style)
ZOO_QUAD_FRAC_EXPORT = 0.5

# ---- temperature (Cloern 2018) — growth + grazing only ----------------------
Q10_GROW  = 1.62
Q10_GRAZE = 2.48
T_REF     = 20.0          # °C (allometry reference)
T_DEFAULT = 24.0          # °C placeholder (override per regime)

# ---- detritus ---------------------------------------------------------------
K_REMIN   = 0.1           # remineralisation rate [d-1] (warm tropical)
W_SINK    = 5.0           # m/d (detritus sinking velocity)

# ---- supply (Stock 2008) — override per regime ------------------------------
FN_DEFAULT = 2.67         # mmol N m-2 d-1 ('all'-regime mean)
DE_DEFAULT = 50.0         # m
# d_e is a BROADCAST parameter (single source of truth): set it once on Inflow
# (with de_label='de') and DetritusSink foreign-references it, so overriding
# Inflow__de per regime drives BOTH F_N/d_e and w_sink/d_e. No separate sinking
# rate to keep in sync.

# ---- fish (top-down lever) — Rykaczewski kernel, scalar rate -----------------
FISH_RATE = 0.0           # default = no-fish baseline; scan up for the crosswise
kernel_P_fish, kernel_Z_fish = compute_fish_kernel_vdl_joint(phyto_esd, zoo_esd)

# ---- omnivory kernel (Mattern Gaussian σ=0.15) ------------------------------
phiPZ_omni = compute_grazing_kernel(phyto_esd, zoo_esd, mode='omni',
                                    theta_opt=THETA_OPT, sigma_log=SIGMA,
                                    convention='mattern')

# ---- initial conditions -----------------------------------------------------
N_INIT = 1.0; P_INIT = 1e-3; Z_INIT = 1e-3; D_INIT = 1e-3
phyto_init = np.full(N_CLASSES, P_INIT)
zoo_init   = np.full(N_CLASSES, Z_INIT)

# ---- solver / time ----------------------------------------------------------
# RK45 + relaxed atol only (NOT LSODA). Pass non-default solver_kwargs at run time.
IVP_SOLVER_KWARGS = {'method': 'RK45', 'atol': 1e-6, 'rtol': 1e-4}
IVP_TIME_END   = 5000.0
ivp_time_array = np.arange(0.0, IVP_TIME_END, 1.0)
STAB_TIME = [0.0, 1.0]
SLIM_OUTPUT_VARS = {'Nutrient__value', 'Phytoplankton__biomass',
                    'Zooplankton__biomass', 'Detritus__value'}

# ---- model ------------------------------------------------------------------
model_npzd = xso.create({
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


def make_npzd_input_vars(fish_rate=FISH_RATE, phiPZ=phiPZ_omni,
                         FN=FN_DEFAULT, de=DE_DEFAULT, T=T_DEFAULT,
                         mu_max=mu_max_arr, halfsat=ks_arr, mP=mP_arr):
    """Input-vars for the NPZD baseline. Override fish_rate / FN / de / T per
    regime or scan. `de` is broadcast (single source of truth) — overriding it
    here drives both the supply F_N/d_e and the detritus sinking w_sink/d_e."""
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
                             'phiPZ': phiPZ, 'Imax': Imax_arr, 'KsZ': KsZ_arr,
                             'q10': Q10_GRAZE, 't_ref': T_REF},
        'GrazingRouter':    {'grazed_phyto': 'P', 'grazed_zoo': 'Z',
                             'assimilated_consumer': 'Z', 'egested_detritus': 'D',
                             'excreted_nutrient': 'N', 'gge': GGE_VAL,
                             'frac_D': GRAZE_FRAC_D, 'frac_export': GRAZE_FRAC_EXPORT},
        'PhytoMortality':   {'population': 'P', 'detritus': 'D', 'nutrient': 'N',
                             'rate': mP, 'frac_D': PHYTO_MORT_FRAC_D,
                             'frac_export': PHYTO_MORT_FRAC_EXPORT},
        'ZooLinMortality':  {'population': 'Z', 'detritus': 'D', 'nutrient': 'N',
                             'rate': M_ZLIN, 'frac_D': ZOO_LIN_FRAC_D,
                             'frac_export': ZOO_LIN_FRAC_EXPORT},
        'ZooQuadMortality': {'population': 'Z', 'detritus': 'D', 'nutrient': 'N',
                             'rate': M_Z, 'frac_D': ZOO_QUAD_FRAC_D,
                             'frac_export': ZOO_QUAD_FRAC_EXPORT},
        'DetritusRemin':    {'detritus': 'D', 'nutrient': 'N', 'k_remin': K_REMIN},
        'DetritusSink':     {'detritus': 'D', 'w_sink': W_SINK, 'de': 'de'},
        'FishGrazing':      {'phyto': 'P', 'zoo': 'Z',
                             'kernel_P': kernel_P_fish, 'kernel_Z': kernel_Z_fish,
                             'rate': fish_rate},
    }


# ---- setup objects ----------------------------------------------------------
model_setup_npzd = xso.setup(
    solver='solve_ivp', model=model_npzd, time=ivp_time_array,
    input_vars=make_npzd_input_vars(), solver_kwargs=IVP_SOLVER_KWARGS)

model_setup_npzd_slim = xso.setup(
    solver='solve_ivp', model=model_npzd, time=ivp_time_array,
    input_vars=make_npzd_input_vars(), output_vars=SLIM_OUTPUT_VARS,
    solver_kwargs=IVP_SOLVER_KWARGS)

model_setup_npzd_stability = xso.setup(
    solver='stability', model=model_npzd, time=STAB_TIME,
    input_vars=make_npzd_input_vars())

# ---- Taniguchi-growth setups (go-forward; all Banas setups above unchanged) --
# Taniguchi μ/K_s (+ consistent Banas-rule mortality mP_tani). The mortality scan
# overrides PhytoMortality__rate per literature treatment; the default here is the
# consistent 0.1·Tani-μ. `model_setup_npzd_tani_scan` adds a slim output + a
# loosened instability guard so parscan cells don't NaN on small negative dips.
_TANI = dict(mu_max=mu_max_tani, halfsat=ks_tani, mP=mP_tani)

model_setup_npzd_tani = xso.setup(
    solver='solve_ivp', model=model_npzd, time=ivp_time_array,
    input_vars=make_npzd_input_vars(**_TANI), solver_kwargs=IVP_SOLVER_KWARGS)

model_setup_npzd_tani_slim = xso.setup(
    solver='solve_ivp', model=model_npzd, time=ivp_time_array,
    input_vars=make_npzd_input_vars(**_TANI), output_vars=SLIM_OUTPUT_VARS,
    solver_kwargs=IVP_SOLVER_KWARGS)

model_setup_npzd_tani_scan = xso.setup(
    solver='solve_ivp', model=model_npzd, time=ivp_time_array,
    input_vars=make_npzd_input_vars(**_TANI), output_vars=SLIM_OUTPUT_VARS,
    solver_kwargs={**IVP_SOLVER_KWARGS, 'instability_neg_threshold': -1e-3})

# Long-time variant for stability/persistence checks (4× longer integration)
ivp_time_array_long = np.arange(0.0, 20000.0, 1.0)
model_setup_npzd_tani_scan_long = xso.setup(
    solver='solve_ivp', model=model_npzd, time=ivp_time_array_long,
    input_vars=make_npzd_input_vars(**_TANI), output_vars=SLIM_OUTPUT_VARS,
    solver_kwargs={**IVP_SOLVER_KWARGS, 'instability_neg_threshold': -1e-3})


# ---- Resolution-variant builder (for sensitivity tests) ---------------------
def _build_grid_arrays(N_classes):
    """Build all size-dependent arrays for an alternative resolution.
    Returns a dict consumed by _make_input_vars_at_resolution."""
    phyto_ = np.logspace(np.log10(PHYTO_ESD_MIN), np.log10(PHYTO_ESD_MAX), N_classes)
    zoo_   = ZOO_PHYTO_RATIO * phyto_
    mu_max_ = 1.36 * phyto_ ** -0.16
    ks_     = 0.33 * phyto_ **  0.48
    mP_     = 0.1 * mu_max_
    zvol_   = (np.pi / 6.0) * zoo_ ** 3
    Imax_   = np.where(zoo_ <= 30.0, 9.8, 30.9 * zvol_ ** -0.16)
    KsZ_    = np.full(N_classes, K_SZ)
    phiPZ_  = compute_grazing_kernel(phyto_, zoo_, mode='omni',
                                     sigma_log=0.15, theta_opt=10.0)
    kP_, kZ_ = compute_fish_kernel_vdl_joint(phyto_, zoo_)
    return dict(N=N_classes, phyto_esd=phyto_, zoo_esd=zoo_,
                mu_max=mu_max_, halfsat=ks_, mP=mP_,
                Imax=Imax_, KsZ=KsZ_, phiPZ=phiPZ_,
                phyto_init=np.full(N_classes, P_INIT),
                zoo_init=np.full(N_classes, Z_INIT),
                kernel_P_fish=kP_, kernel_Z_fish=kZ_)

def _make_input_vars_at_resolution(grids, fish_rate=FISH_RATE,
                                    FN=FN_DEFAULT, de=DE_DEFAULT, T=T_DEFAULT):
    """Build input_vars dict from a resolution-specific grid dict."""
    return {
        'Nutrient':         {'value_label': 'N', 'value_init': N_INIT},
        'Phytoplankton':    {'biomass_label': 'P', 'biomass_init': grids['phyto_init'],
                             'phyto_esd_index': grids['phyto_esd'].tolist(),
                             'phyto_esd_label': 'phyto_esd'},
        'Zooplankton':      {'biomass_label': 'Z', 'biomass_init': grids['zoo_init'],
                             'zoo_esd_index': grids['zoo_esd'].tolist(),
                             'zoo_esd_label': 'zoo_esd'},
        'Detritus':         {'value_label': 'D', 'value_init': D_INIT},
        'Inflow':           {'var': 'N', 'FN': FN, 'de': de, 'de_label': 'de'},
        'Temperature':      {'forcing_label': 'temperature', 'value': T},
        'Growth':           {'resource': 'N', 'consumer': 'P', 'temperature': 'temperature',
                             'mu_max_label': 'mu_max',
                             'mu_max': grids['mu_max'], 'halfsat': grids['halfsat'],
                             'q10': Q10_GROW, 't_ref': T_REF},
        'Grazing':          {'resource': 'P', 'consumer': 'Z', 'temperature': 'temperature',
                             'phiPZ': grids['phiPZ'], 'Imax': grids['Imax'],
                             'KsZ': grids['KsZ'],
                             'q10': Q10_GRAZE, 't_ref': T_REF},
        'GrazingRouter':    {'grazed_phyto': 'P', 'grazed_zoo': 'Z',
                             'assimilated_consumer': 'Z', 'egested_detritus': 'D',
                             'excreted_nutrient': 'N', 'gge': GGE_VAL,
                             'frac_D': GRAZE_FRAC_D, 'frac_export': GRAZE_FRAC_EXPORT},
        'PhytoMortality':   {'population': 'P', 'detritus': 'D', 'nutrient': 'N',
                             'rate': grids['mP'], 'frac_D': PHYTO_MORT_FRAC_D,
                             'frac_export': PHYTO_MORT_FRAC_EXPORT},
        'ZooLinMortality':  {'population': 'Z', 'detritus': 'D', 'nutrient': 'N',
                             'rate': M_ZLIN, 'frac_D': ZOO_LIN_FRAC_D,
                             'frac_export': ZOO_LIN_FRAC_EXPORT},
        'ZooQuadMortality': {'population': 'Z', 'detritus': 'D', 'nutrient': 'N',
                             'rate': M_Z, 'frac_D': ZOO_QUAD_FRAC_D,
                             'frac_export': ZOO_QUAD_FRAC_EXPORT},
        'DetritusRemin':    {'detritus': 'D', 'nutrient': 'N', 'k_remin': K_REMIN},
        'DetritusSink':     {'detritus': 'D', 'w_sink': W_SINK, 'de': 'de'},
        'FishGrazing':      {'phyto': 'P', 'zoo': 'Z',
                             'kernel_P': grids['kernel_P_fish'],
                             'kernel_Z': grids['kernel_Z_fish'],
                             'rate': fish_rate},
    }

# Pre-built resolution variants — baseline N_CLASSES = 40 is unchanged.
_GRIDS_LORES = _build_grid_arrays(20)
_GRIDS_HIRES = _build_grid_arrays(80)

model_setup_npzd_tani_scan_lores = xso.setup(
    solver='solve_ivp', model=model_npzd, time=ivp_time_array,
    input_vars=_make_input_vars_at_resolution(_GRIDS_LORES),
    output_vars=SLIM_OUTPUT_VARS,
    solver_kwargs={**IVP_SOLVER_KWARGS, 'instability_neg_threshold': -1e-3})

model_setup_npzd_tani_scan_hires = xso.setup(
    solver='solve_ivp', model=model_npzd, time=ivp_time_array,
    input_vars=_make_input_vars_at_resolution(_GRIDS_HIRES),
    output_vars=SLIM_OUTPUT_VARS,
    solver_kwargs={**IVP_SOLVER_KWARGS, 'instability_neg_threshold': -1e-3})

# Expose the variant grids so notebooks can read phyto_esd / band masks for plotting
phyto_esd_lores = _GRIDS_LORES['phyto_esd']
zoo_esd_lores   = _GRIDS_LORES['zoo_esd']
phyto_esd_hires = _GRIDS_HIRES['phyto_esd']
zoo_esd_hires   = _GRIDS_HIRES['zoo_esd']


model_setup_npzd_tani_stability = xso.setup(
    solver='stability', model=model_npzd, time=STAB_TIME,
    input_vars=make_npzd_input_vars(**_TANI))

# ---- Per-class quadratic Z closure variant ----------------------------------
# Identical to model_npzd except the quadratic zoo closure uses m_Z·Z_j²
# (each class self-regulates) instead of m_Z·Z_j·ΣZ (bulk coupling). Tests
# whether the omnivory-driven mcs descending limb depends on inter-class
# coupling through ΣZ. Paired with Taniguchi growth in the *_tani_perclassZ_*
# setups below.
model_npzd_perclassZ = xso.create({
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
    'ZooQuadMortality': ZooQuadraticMortality_perclass_route,  # <-- only change
    'DetritusRemin':    DetritusRemineralization,
    'DetritusSink':     DetritusSinking,
    'FishGrazing':      FishGrazing_Kernel_rate,
})

model_setup_npzd_tani_perclassZ = xso.setup(
    solver='solve_ivp', model=model_npzd_perclassZ, time=ivp_time_array,
    input_vars=make_npzd_input_vars(**_TANI), solver_kwargs=IVP_SOLVER_KWARGS)

model_setup_npzd_tani_perclassZ_slim = xso.setup(
    solver='solve_ivp', model=model_npzd_perclassZ, time=ivp_time_array,
    input_vars=make_npzd_input_vars(**_TANI), output_vars=SLIM_OUTPUT_VARS,
    solver_kwargs=IVP_SOLVER_KWARGS)

model_setup_npzd_tani_perclassZ_scan = xso.setup(
    solver='solve_ivp', model=model_npzd_perclassZ, time=ivp_time_array,
    input_vars=make_npzd_input_vars(**_TANI), output_vars=SLIM_OUTPUT_VARS,
    solver_kwargs={**IVP_SOLVER_KWARGS, 'instability_neg_threshold': -1e-3})

# ---- Banas-mortality model variant (proper foreign-μ_max mortality) ----------
# Identical to model_npzd except PhytoMortality is the Banas form: m_P = coeff·μ_max·P
# with μ_max foreign-referenced from Growth (so it tracks the actual growth allometry).
# coeff is a SCALAR (PhytoMortality__coeff) -> a clean parscan axis for the mortality
# sweep. Paired with Taniguchi growth below.
model_npzd_banas = xso.create({
    'Nutrient':         Nutrient,
    'Phytoplankton':    PhytoSizeSpectrum,
    'Zooplankton':      ZooSizeSpectrum,
    'Detritus':         Detritus,
    'Inflow':           StockNutrientSupply,
    'Temperature':      ConstantTemperatureForcing,
    'Growth':           MonodGrowth_T,
    'Grazing':          DistributedGrazing_TypeIII_T,
    'GrazingRouter':    DistributedGrazingRouter_route,
    'PhytoMortality':   BanasPhytoMortality_route,    # <-- only change vs model_npzd
    'ZooLinMortality':  ZooLinearMortality_route,
    'ZooQuadMortality': ZooQuadraticMortality_route,
    'DetritusRemin':    DetritusRemineralization,
    'DetritusSink':     DetritusSinking,
    'FishGrazing':      FishGrazing_Kernel_rate,
})


def make_npzd_banas_input_vars(coeff=BANAS_MORT_COEFF, **kw):
    """Input-vars for model_npzd_banas: same as make_npzd_input_vars but the
    PhytoMortality slot is the Banas form (foreign μ_max + scalar coeff). Pass
    growth via mu_max=/halfsat= (e.g. the Taniguchi arrays). The `mP=` kwarg of
    make_npzd_input_vars is ignored here (the whole PhytoMortality slot is replaced)."""
    iv = make_npzd_input_vars(**kw)
    iv['PhytoMortality'] = {'population': 'P', 'detritus': 'D', 'nutrient': 'N',
                            'mu_max': 'mu_max', 'coeff': coeff,
                            'frac_D': PHYTO_MORT_FRAC_D, 'frac_export': PHYTO_MORT_FRAC_EXPORT}
    return iv


_TANI_BANAS = dict(mu_max=mu_max_tani, halfsat=ks_tani)   # growth only; mortality = coeff·μ_max

model_setup_npzd_banas = xso.setup(
    solver='solve_ivp', model=model_npzd_banas, time=ivp_time_array,
    input_vars=make_npzd_banas_input_vars(**_TANI_BANAS), solver_kwargs=IVP_SOLVER_KWARGS)

model_setup_npzd_banas_slim = xso.setup(
    solver='solve_ivp', model=model_npzd_banas, time=ivp_time_array,
    input_vars=make_npzd_banas_input_vars(**_TANI_BANAS), output_vars=SLIM_OUTPUT_VARS,
    solver_kwargs=IVP_SOLVER_KWARGS)

model_setup_npzd_banas_scan = xso.setup(
    solver='solve_ivp', model=model_npzd_banas, time=ivp_time_array,
    input_vars=make_npzd_banas_input_vars(**_TANI_BANAS), output_vars=SLIM_OUTPUT_VARS,
    solver_kwargs={**IVP_SOLVER_KWARGS, 'instability_neg_threshold': -1e-3})
