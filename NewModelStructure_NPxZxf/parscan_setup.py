"""
NPxZxF Parscan Setup
====================
Minimal model file for use with run_xso_parscan / run_xso_stabilityscan.
Exports: model, model_setup, phyto_esd, zoo_esd

This is a near-duplicate of npxzxf_models.py, with the addition of
the model_setup object that the parscan workers need to import.
"""

import numpy as np
import xso

from npxzxf_comps import (
    Nutrient, PhytoSizeSpectrum, ZooSizeSpectrum,
    ConstantExternalNutrient, ConstantFishForcing,
    LinearForcingInput, MonodGrowth_SizeBased,
    SizebasedGrazingMatrix_Full_TypeIII, GGE_Full_SizeDep,
    PhytoMortality_toN, ZooQuadraticMortality,
    FishGrazing_SizeBased,
)


# =============================================================================
# ALLOMETRIC FUNCTIONS (copied from npxzxf_models.py)
# =============================================================================

def generate_size_classes(n, esd_min, esd_max):
    return np.logspace(np.log10(esd_min), np.log10(esd_max), n)

def compute_mu_max_maranon(esd):
    mu_max = np.zeros_like(esd)
    small = esd <= 5.38
    mu_max[small] = 0.33 * esd[small] ** 0.57
    large = esd > 5.38
    mu_max[large] = 1.83 * esd[large] ** (-0.45)
    mu_max[esd > 20.0] *= 1.5
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

def _ramp(esd, esd_min, preference):
    w = np.zeros_like(esd)
    mask = esd >= esd_min
    if np.any(mask):
        esd_max = np.max(esd)
        if esd_max > esd_min:
            w[mask] = preference * (esd[mask] - esd_min) / (esd_max - esd_min)
        else:
            w[mask] = preference
    return w

def compute_fish_weights_P(phyto_esd, p_esd_min=100.0, p_preference=0.3):
    return _ramp(phyto_esd, p_esd_min, p_preference)

def compute_fish_weights_Z(zoo_esd, z_esd_min=30.0, z_preference=1.0):
    return _ramp(zoo_esd, z_esd_min, z_preference)


# =============================================================================
# SIZE CLASSES & PARAMETERS
# =============================================================================
n_classes = 15
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
gge   = compute_gge(zoo_esd)
m_Z   = 0.1
KsZ   = 3.0

phiPZ = compute_grazing_kernel(phyto_esd, zoo_esd)

w_P = compute_fish_weights_P(phyto_esd)
w_Z = compute_fish_weights_Z(zoo_esd)
fish_biomass = 1.0
fish_rate    = 0.005

N0_cariaco       = 5.5564
dilution_cariaco = 0.016786


# =============================================================================
# BUILD MODEL
# =============================================================================
model = xso.create({
    'Nutrient':       Nutrient,
    'Phytoplankton':  PhytoSizeSpectrum,
    'Zooplankton':    ZooSizeSpectrum,
    'N0':             ConstantExternalNutrient,
    'Inflow':         LinearForcingInput,
    'Growth':         MonodGrowth_SizeBased,
    'Grazing':        SizebasedGrazingMatrix_Full_TypeIII,
    'GGE':            GGE_Full_SizeDep,
    'PhytoMortality': PhytoMortality_toN,
    'ZooMortality':   ZooQuadraticMortality,
    'FishForcing':    ConstantFishForcing,
    'FishGrazing':    FishGrazing_SizeBased,
})

# =============================================================================
# INPUT DICTIONARY
# =============================================================================
input_vars = {
    'Nutrient':      {'value_label': 'N', 'value_init': N_init},
    'Phytoplankton': {'biomass_label': 'P', 'biomass_init': phyto_init,
                      'phyto_index': phyto_esd.tolist()},
    'Zooplankton':   {'biomass_label': 'Z', 'biomass_init': zoo_init,
                      'zoo_index': zoo_esd.tolist()},
    'N0':     {'forcing_label': 'N0', 'value': N0_cariaco},
    'Inflow': {'forcing': 'N0', 'rate': dilution_cariaco, 'var': 'N'},
    'Growth': {'resource': 'N', 'consumer': 'P',
               'halfsat': K_s, 'mu_max': mu_max},
    'Grazing': {'resource': 'P', 'consumer': 'Z',
                'phiPZ': phiPZ, 'Imax': I_max, 'KsZ': KsZ},
    'GGE':     {'grazed_phyto': 'P', 'grazed_zoo': 'Z',
                'assimilated_consumer': 'Z', 'recycled_nutrient': 'N',
                'gge': gge},
    'PhytoMortality': {'population': 'P', 'nutrient': 'N',
                       'rate': m_P, 'recycle_frac': m_P_recycled},
    'ZooMortality':   {'population': 'Z', 'rate': m_Z},
    'FishForcing': {'forcing_label': 'F_forcing', 'value': fish_biomass},
    'FishGrazing': {'phyto': 'P', 'zoo': 'Z',
                    'fish_forcing': 'F_forcing',
                    'w_P': w_P, 'w_Z': w_Z, 'rate': fish_rate},
}

# =============================================================================
# MODEL SETUP (this is what parscan workers import)
# =============================================================================
model_setup = xso.setup(
    solver='solve_ivp',
    model=model,
    time=np.arange(0, 5000, 1),
    input_vars=input_vars,
)