"""
NPxZxF Parscan Setup
====================
Minimal model file for use with run_xso_parscan / run_xso_stabilityscan.
Exports: model, model_setup, phyto_esd, zoo_esd
"""

import numpy as np
import xso

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from xso.parscans import avg_tail

from cariaco_ssm_comps import (
    Nutrient, PhytoSizeSpectrum, ZooSizeSpectrum,
    StockNutrientSupply, ConstantFishForcing,
    LinearForcingInput, MonodGrowth_SizeBased,
    SizebasedGrazingMatrix_Full_TypeIII,
    FishGrazing_Kernel,
    Detritus,
    GGE_Full_withD,
    PhytoMortality_toD_toN,
    PhytoQuadraticMortality_perclass_toD,
    PhytoQuadraticMortality_bulk_toD,
    ZooQuadraticMortality_toD,
    DetritusRemineralization,
    DetritusSinking,
)


# =============================================================================
# ALLOMETRIC FUNCTIONS 
# =============================================================================

def generate_size_classes(n, esd_min, esd_max):
    return np.logspace(np.log10(esd_min), np.log10(esd_max), n)

def compute_mu_max_maranon(esd):
    mu_max = np.zeros_like(esd)
    small = esd <= 5.38
    mu_max[small] = 0.33 * esd[small] ** 0.57
    large = esd > 5.38
    mu_max[large] = 1.83 * esd[large] ** (-0.45)
    #mu_max[esd > 20.0] *= 1.5
    return mu_max

def compute_K_s(esd):
    return 0.144 * esd ** 0.81

def compute_I_max(esd):
    return 26.0 * esd ** (-0.48)

def compute_gge(esd):
    frac = (np.log10(esd) - np.log10(5.0)) / (np.log10(2000.0) - np.log10(5.0))
    frac = np.clip(frac, 0.0, 1.0)
    return 0.35 + (0.15 - 0.35) * frac

def compute_grazing_kernel(phyto_esd, zoo_esd, theta_opt=10.0, sigma_log=0.25):
    prey_esd = np.concatenate([phyto_esd, zoo_esd])
    n_P = len(phyto_esd)
    n_Z = len(zoo_esd)
    log_ratio = np.log10(zoo_esd[None, :] / prey_esd[:, None])
    log_theta = np.log10(theta_opt)
    phiPZ = np.exp(-((log_ratio - log_theta) ** 2) / (2 * sigma_log ** 2))
    for j in range(n_Z):
        phiPZ[n_P + j, j] = 0.0
    return phiPZ



def clearance_rate_sardine_vdl(prey_length_um, filter_feeding=False):
    """Sardine size-specific clearance rate from Rykaczewski (2019), Eq. 3.

    This is Rykaczewski's modification of van der Lingen (1999), Eq. 5,
    fit to laboratory clearance-rate measurements on adult Sardinops sagax.
    The functional form is a sum of two logistic (sigmoid) terms, one
    centered at 15 µm (fine filter-feeding mechanism) and one centered at
    800 µm (coarser retention mechanism).

    In filter-feeding mode, the clearance rate is clamped flat above
    1230 µm — the value at 1230 µm is used for all larger prey. This
    reflects the biological reality that filter feeding rate is limited
    by water flow through the gill rakers, not by prey size, once prey
    are large enough to be reliably retained.

    Parameters
    ----------
    prey_length_um : array_like
        Prey length in µm. For phytoplankton, this is the longest cell
        dimension; for copepods, this is prosome length (NOT ESD —
        copepod prosome length is typically ~2–3× the ESD of an
        equivalent sphere, so apply a length-conversion factor when
        passing zooplankton ESDs from your size-spectrum grid).
    filter_feeding : bool
        If True (default), apply the >1230 µm clamp for filter feeding.
        If False, return the raw equation value (use with caution
        above ~2.7 mm, where the underlying fit is unreliable).

    Returns
    -------
    F_S : ndarray
        Size-specific clearance rate, same shape as input. Units are
        l fish^-1 min^-1 per prey size class as defined in the source
        paper. For use as a peak-normalized kernel, divide by the
        max value across your grid.

    References
    ----------
    Rykaczewski, R. R. (2019). Changes in mesozooplankton size structure
        along a trophic gradient in the California Current Ecosystem and
        implications for planktivorous fishes. Marine Ecology Progress
        Series, 617–618, 165–182. (Eq. 3)
    van der Lingen, C. D. (1999). The feeding ecology of, and carbon and
        nitrogen budgets for, sardine Sardinops sagax in the southern
        Benguela upwelling system. PhD dissertation, University of Cape
        Town. (Original Eq. 5, which Rykaczewski's Eq. 3 modifies.)
    van der Lingen, C. D. (1994). Effect of particle size and concentration
        on the feeding behaviour of adult pilchard Sardinops sagax.
        Marine Ecology Progress Series, 109, 1–13. (Underlying lab data.)
    """
    x = np.asarray(prey_length_um, dtype=float)

    def _f(xv):
        # First logistic term: fine-filter mechanism, centered at 15 µm
        e1 = np.exp(0.0198 * (xv - 15.0))
        term1 = (9.03 * e1) / (12.03 + 0.75 * e1)

        # Second logistic term: coarser retention, centered at 800 µm
        e2 = np.exp(0.00843 * (xv - 800.0))
        term2 = (9.96 * e2) / (30.8 + 0.323 * e2)

        return term1 + term2

    F_S = _f(x)

    if filter_feeding:
        # Clamp flat above 1230 µm: use the value at 1230 µm everywhere
        # the prey is larger
        F_S_at_1230 = _f(np.array(1230.0))
        F_S = np.where(x > 1230.0, F_S_at_1230, F_S)

    return F_S


def compute_fish_kernel_vdl_joint(phyto_esd, zoo_esd):
    """Sardine feeding kernel on P and Z grids, jointly peak-normalized.

    Evaluates the Rykaczewski (2019) Eq. 3 clearance-rate curve on both
    the phytoplankton and zooplankton ESD grids, then normalizes both
    by the same maximum value so that peak = 1 occurs on whichever grid
    contains the absolute maximum of the curve (typically the zoo grid,
    since the curve peaks near 1230 µm).

    Joint normalization preserves the *relative* weighting between P and
    Z predicted by the clearance-rate curve. Normalizing each grid
    independently would artificially boost the phyto kernel to peak = 1
    even though adult sardines clear large zooplankton much more
    efficiently than small phytoplankton.

    Parameters
    ----------
    phyto_esd : array_like
        Phytoplankton size-class ESDs in µm.
    zoo_esd : array_like
        Zooplankton size-class ESDs in µm. For strict consistency with
        Rykaczewski's equation (which takes copepod prosome length),
        multiply by ~2.5 before passing if you want to correct for
        the length-vs-ESD mismatch.

    Returns
    -------
    kernel_P : ndarray
        Selectivity weights on the phyto grid, same shape as phyto_esd.
    kernel_Z : ndarray
        Selectivity weights on the zoo grid, same shape as zoo_esd.

    Both kernels share a single normalization constant, so the `rate`
    parameter in FishGrazing_Lognormal retains its meaning as the peak
    mass-specific grazing rate per unit fish biomass at the overall
    preferred prey size.

    See Also
    --------
    clearance_rate_sardine_vdl : underlying two-sigmoid curve
    compute_fish_kernel_lognormal : alternative symmetric log-normal kernel
    """
    F_P = clearance_rate_sardine_vdl(phyto_esd)
    F_Z = clearance_rate_sardine_vdl(zoo_esd)
    F_max = max(F_P.max(), F_Z.max())
    return F_P / F_max, F_Z / F_max

    
# =============================================================================
# SIZE CLASSES & PARAMETERS
# =============================================================================
n_classes = 12
phyto_esd = generate_size_classes(n_classes, esd_min=0.5, esd_max=200)
zoo_esd   = generate_size_classes(n_classes, esd_min=5, esd_max=2000)

phyto_init = np.full(n_classes, 0.01)
zoo_init   = np.full(n_classes, 0.001)
N_init     = 0.1

mu_max       = compute_mu_max_maranon(phyto_esd)
K_s          = compute_K_s(phyto_esd)
m_P          = 0.1 * mu_max
m_P_recycled = 1.0

I_max = compute_I_max(zoo_esd)
gge   = 0.25
m_Z   = 0.1
KsZ   = 3.0

#phiPZ = compute_grazing_kernel(phyto_esd, zoo_esd)

fish_biomass = 1.0
fish_rate    = 0.005
# data-based kernel:
kernel_P_fish,kernel_Z_fish = compute_fish_kernel_vdl_joint(phyto_esd,zoo_esd)


F_N = 2.6695
d_e = 50.0                     # m, surface box depth

# Parameters for Detritus
D_init = 0.01                  # mmol N m-3
f_egest_D_zoo = 0.75           # Fasham: 75% egesta to D, 25% sloppy to N
f_mort_D_phyto = 0.9           # Fasham-style: most mortality to D
f_mort_D_zoo = 0.5             # Stock-style: half to D, half exported
k_remin = 0.1                  # d-1, warm tropical
w_sink = 5.0                   # m d-1, bulk detritus sinking
sinking_rate = w_sink / d_e    # d-1


# =============================================================================
# BUILD MODEL
# =============================================================================
# Model dict
model = xso.create({
    'Nutrient':       Nutrient,
    'Phytoplankton':  PhytoSizeSpectrum,
    'Zooplankton':    ZooSizeSpectrum,
    'Detritus':       Detritus,
    'Inflow':         StockNutrientSupply,
    'Growth':         MonodGrowth_SizeBased,
    'Grazing':        SizebasedGrazingMatrix_Full_TypeIII,
    'GGE':            GGE_Full_withD,
    'PhytoMortality': PhytoMortality_toD_toN,
    'ZooMortality':   ZooQuadraticMortality_toD,
    'DetritusRemin':  DetritusRemineralization,
    'DetritusSink':   DetritusSinking,
    'FishForcing':    ConstantFishForcing,
    'FishGrazing':    FishGrazing_Kernel,
})



# =============================================================================
# INPUT DICTIONARY
# =============================================================================
input_vars = {
    'Nutrient':      {'value_label': 'N', 'value_init': N_init},
    'Phytoplankton': {'biomass_label': 'P', 'biomass_init': phyto_init,
                      'phyto_esd_index': phyto_esd.tolist(),
                      'phyto_esd_label': 'phyto_esd'},
    'Zooplankton':   {'biomass_label': 'Z', 'biomass_init': zoo_init,
                      'zoo_esd_index': zoo_esd.tolist(),
                      'zoo_esd_label': 'zoo_esd'},
    'Detritus': {'value_label': 'D', 'value_init': D_init},
    'Inflow': {'var': 'N', 'FN': F_N, 'de':d_e},
    'Growth': {'resource': 'N', 'consumer': 'P',
               'halfsat': K_s, 'mu_max': mu_max},
    'Grazing': {'resource': 'P', 'consumer': 'Z',
                'phyto_esd': 'phyto_esd',
                'zoo_esd': 'zoo_esd',
                'theta_opt': 10.0,
                'sigma_log': 0.25,
                'Imax': I_max, 'KsZ': KsZ},
    'GGE': {'grazed_phyto': 'P', 'grazed_zoo': 'Z',
            'assimilated_consumer': 'Z',
            'egested_detritus': 'D',
            'excreted_nutrient': 'N',
            'gge': gge, 'f_egest_D': f_egest_D_zoo},
    'PhytoMortality': {'population': 'P', 'detritus': 'D', 'nutrient': 'N',
                       'rate': m_P, 'f_mort_D': f_mort_D_phyto},
    'ZooMortality': {'population': 'Z', 'detritus': 'D',
                     'rate': m_Z, 'f_mort_D': f_mort_D_zoo},
    'FishForcing': {'forcing_label': 'F_forcing', 'value': fish_biomass},
    'FishGrazing': {'phyto': 'P', 'zoo': 'Z',
                'fish_forcing': 'F_forcing',
                'kernel_P': kernel_P_fish,
                'kernel_Z': kernel_Z_fish,
                'rate': fish_rate},
    'DetritusRemin': {'detritus': 'D', 'nutrient': 'N', 'k_remin': k_remin},
    'DetritusSink':  {'detritus': 'D', 'sinking_rate': sinking_rate},
}

# =============================================================================
# MODEL SETUP (this is what parscan workers import)
# =============================================================================
model_setup = xso.setup(
    solver='solve_ivp',
    model=model,
    time=np.arange(0, 5000, 1),
    input_vars=input_vars
)

model_setup_slim = xso.setup(
    solver='solve_ivp',
    model=model,
    time=np.arange(0, 5000, 1),
    input_vars=input_vars,
    output_vars= {'Phytoplankton__biomass', 'Zooplankton__biomass', 'Nutrient__value',
              'Detritus__value', 'DetritusSink__sinking_value',
              'Growth__uptake_value', 'Inflow__de'}

)

model_setup_stability = xso.setup(
    solver='stability',
    model=model,
    time=[0,1],
    input_vars=input_vars
)


# =============================================================================
# GRID VARIANTS — floor (0.5→0.2 µm) × resolution (12/40/80) difference-tests
# (added 2026-06-15). Same `model` + Marañón allometries + scalar defaults;
# ONLY the phyto grid floor and class count change. Growth can still be
# overridden to Taniguchi at scan time via Growth__mu_max / Growth__halfsat /
# PhytoMortality__rate (arrays computed on the matching exported *_phyto_esd_*).
# input_vars is helper-built (cf. baseline_r0_setups.make_baseline_input_vars);
# each named setup below is spelled out explicitly.
# =============================================================================

_SLIM_OUT = {'Phytoplankton__biomass', 'Zooplankton__biomass', 'Nutrient__value',
             'Detritus__value', 'DetritusSink__sinking_value',
             'Growth__uptake_value', 'Inflow__de'}

def make_ssm_input_vars(n_cls, esd_min_phyto):
    """cariaco_ssm input_vars on a phyto grid [esd_min_phyto, 200] µm, n_cls
    log-spaced classes (zoo = 10× phyto). Marañón growth, old I_max, fish ON,
    scalar defaults identical to the module baseline. Returns (p_esd, z_esd, iv)."""
    p_esd = generate_size_classes(n_cls, esd_min_phyto, 200.0)
    z_esd = generate_size_classes(n_cls, 10.0 * esd_min_phyto, 2000.0)
    mu = compute_mu_max_maranon(p_esd)
    ks = compute_K_s(p_esd)
    mp = 0.1 * mu
    im = compute_I_max(z_esd)
    kP, kZ = compute_fish_kernel_vdl_joint(p_esd, z_esd)
    iv = {
        'Nutrient':       {'value_label': 'N', 'value_init': N_init},
        'Phytoplankton':  {'biomass_label': 'P', 'biomass_init': np.full(n_cls, 0.01),
                           'phyto_esd_index': p_esd.tolist(), 'phyto_esd_label': 'phyto_esd'},
        'Zooplankton':    {'biomass_label': 'Z', 'biomass_init': np.full(n_cls, 0.001),
                           'zoo_esd_index': z_esd.tolist(), 'zoo_esd_label': 'zoo_esd'},
        'Detritus':       {'value_label': 'D', 'value_init': D_init},
        'Inflow':         {'var': 'N', 'FN': F_N, 'de': d_e},
        'Growth':         {'resource': 'N', 'consumer': 'P', 'halfsat': ks, 'mu_max': mu},
        'Grazing':        {'resource': 'P', 'consumer': 'Z',
                           'phyto_esd': 'phyto_esd', 'zoo_esd': 'zoo_esd',
                           'theta_opt': 10.0, 'sigma_log': 0.25, 'Imax': im, 'KsZ': KsZ},
        'GGE':            {'grazed_phyto': 'P', 'grazed_zoo': 'Z', 'assimilated_consumer': 'Z',
                           'egested_detritus': 'D', 'excreted_nutrient': 'N',
                           'gge': gge, 'f_egest_D': f_egest_D_zoo},
        'PhytoMortality': {'population': 'P', 'detritus': 'D', 'nutrient': 'N',
                           'rate': mp, 'f_mort_D': f_mort_D_phyto},
        'ZooMortality':   {'population': 'Z', 'detritus': 'D', 'rate': m_Z, 'f_mort_D': f_mort_D_zoo},
        'FishForcing':    {'forcing_label': 'F_forcing', 'value': fish_biomass},
        'FishGrazing':    {'phyto': 'P', 'zoo': 'Z', 'fish_forcing': 'F_forcing',
                           'kernel_P': kP, 'kernel_Z': kZ, 'rate': fish_rate},
        'DetritusRemin':  {'detritus': 'D', 'nutrient': 'N', 'k_remin': k_remin},
        'DetritusSink':   {'detritus': 'D', 'sinking_rate': sinking_rate},
    }
    return p_esd, z_esd, iv

def _ssm_slim_setup(n_cls, esd_min_phyto):
    _, _, iv = make_ssm_input_vars(n_cls, esd_min_phyto)
    return xso.setup(solver='solve_ivp', model=model, time=np.arange(0, 5000, 1),
                     input_vars=iv, output_vars=_SLIM_OUT)

# Exposed grids (use the one matching the setup you scan, to build Taniguchi overrides)
phyto_esd_f02_n12, zoo_esd_f02_n12, _ = make_ssm_input_vars(12, 0.2)
phyto_esd_f05_n40, zoo_esd_f05_n40, _ = make_ssm_input_vars(40, 0.5)
phyto_esd_f05_n80, zoo_esd_f05_n80, _ = make_ssm_input_vars(80, 0.5)
phyto_esd_f02_n40, zoo_esd_f02_n40, _ = make_ssm_input_vars(40, 0.2)
phyto_esd_f02_n80, zoo_esd_f02_n80, _ = make_ssm_input_vars(80, 0.2)

# Named slim setups (parscan targets) — floor × resolution
model_setup_slim_f02_n12 = _ssm_slim_setup(12, 0.2)   # floor only (vs the 12/0.5 baseline)
model_setup_slim_f05_n40 = _ssm_slim_setup(40, 0.5)   # resolution only
model_setup_slim_f05_n80 = _ssm_slim_setup(80, 0.5)   # resolution only
model_setup_slim_f02_n40 = _ssm_slim_setup(40, 0.2)   # floor + resolution (≈ current grid)
model_setup_slim_f02_n80 = _ssm_slim_setup(80, 0.2)   # floor + resolution


# =============================================================================
# QUADRATIC PHYTO MORTALITY variants (quick test, 2026-06-15) — f02_n40 grid.
# Swap the linear PhytoMortality for a quadratic form (per-class or bulk).
# Growth defaults to Marañón (module); override Growth__mu_max/halfsat for
# Taniguchi at scan time. The quadratic coefficient is the scalar
# PhytoMortality__rate (scan axis for the magnitude test).
# =============================================================================

def _phytoquad_input_vars(quad_rate=0.1):
    """f02_n40 input_vars with the PhytoMortality slot wired for a SCALAR
    quadratic-mortality coefficient (used by both quad components)."""
    _, _, iv = make_ssm_input_vars(40, 0.2)
    iv['PhytoMortality'] = {'population': 'P', 'detritus': 'D', 'nutrient': 'N',
                            'rate': quad_rate, 'f_mort_D': f_mort_D_phyto}
    return iv

model_phytoquad_pc = xso.create({
    'Nutrient': Nutrient, 'Phytoplankton': PhytoSizeSpectrum,
    'Zooplankton': ZooSizeSpectrum, 'Detritus': Detritus,
    'Inflow': StockNutrientSupply, 'Growth': MonodGrowth_SizeBased,
    'Grazing': SizebasedGrazingMatrix_Full_TypeIII, 'GGE': GGE_Full_withD,
    'PhytoMortality': PhytoQuadraticMortality_perclass_toD,   # <- per-class m·P²
    'ZooMortality': ZooQuadraticMortality_toD,
    'DetritusRemin': DetritusRemineralization, 'DetritusSink': DetritusSinking,
    'FishForcing': ConstantFishForcing, 'FishGrazing': FishGrazing_Kernel,
})

model_phytoquad_bulk = xso.create({
    'Nutrient': Nutrient, 'Phytoplankton': PhytoSizeSpectrum,
    'Zooplankton': ZooSizeSpectrum, 'Detritus': Detritus,
    'Inflow': StockNutrientSupply, 'Growth': MonodGrowth_SizeBased,
    'Grazing': SizebasedGrazingMatrix_Full_TypeIII, 'GGE': GGE_Full_withD,
    'PhytoMortality': PhytoQuadraticMortality_bulk_toD,        # <- bulk m·P·ΣP
    'ZooMortality': ZooQuadraticMortality_toD,
    'DetritusRemin': DetritusRemineralization, 'DetritusSink': DetritusSinking,
    'FishForcing': ConstantFishForcing, 'FishGrazing': FishGrazing_Kernel,
})

model_setup_slim_phytoquad_pc = xso.setup(
    solver='solve_ivp', model=model_phytoquad_pc, time=np.arange(0, 5000, 1),
    input_vars=_phytoquad_input_vars(), output_vars=_SLIM_OUT)

model_setup_slim_phytoquad_bulk = xso.setup(
    solver='solve_ivp', model=model_phytoquad_bulk, time=np.arange(0, 5000, 1),
    input_vars=_phytoquad_input_vars(), output_vars=_SLIM_OUT)