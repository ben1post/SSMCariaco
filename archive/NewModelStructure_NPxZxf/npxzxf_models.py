"""
NPxZxF Model Setup for CARIACO Basin
=====================================
Defines allometric functions, computes size-dependent parameters,
builds the XSO model, and exports the input dictionary.

Importable objects:
    model, input_vars, phyto_esd, zoo_esd
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
# ALLOMETRIC FUNCTIONS
# =============================================================================

def generate_size_classes(n, esd_min, esd_max):
    """Generate log-spaced size classes (µm ESD)."""
    return np.logspace(np.log10(esd_min), np.log10(esd_max), n)


# --- Phytoplankton ----------------------------------------------------------

def compute_mu_max_maranon(esd):
    """
    Maximum growth rate (d-1) — Marañón et al. (2013) unimodal.
    Piecewise RMA regressions converted from cell volume to ESD.
    Diatom boost (1.5×) above 20 µm (Mattern 2026 / DARWIN).
    """
    mu_max = np.zeros_like(esd)
    small = esd <= 5.38
    mu_max[small] = 0.33 * esd[small] ** 0.57
    large = esd > 5.38
    mu_max[large] = 1.83 * esd[large] ** (-0.45)
    mu_max[esd > 20.0] *= 1.5
    return mu_max


def compute_K_s(esd):
    """
    Half-saturation for nutrient uptake (mmol N m-3).
    Litchman (2007) / Ward et al. (2012).  ESD exponent 0.81.
    """
    return 0.144 * esd ** 0.81


# --- Zooplankton ------------------------------------------------------------

def compute_I_max(esd):
    """
    Maximum ingestion rate (d-1).
    Hansen (1997) / Banas (2011).  ESD exponent -0.48.
    """
    return 26.0 * esd ** (-0.48)


def compute_gge(esd):
    """
    Gross growth efficiency (dimensionless).
    Linear in log-ESD from 0.35 (5 µm) to 0.15 (2000 µm).
    """
    frac = (np.log10(esd) - np.log10(5.0)) / (np.log10(2000.0) - np.log10(5.0))
    frac = np.clip(frac, 0.0, 1.0)
    return 0.35 + (0.15 - 0.35) * frac


# --- Grazing kernel ---------------------------------------------------------

def compute_grazing_kernel(phyto_esd, zoo_esd, theta_opt=10.0, sigma_log=0.25):
    """
    Log-normal grazing preference matrix φ(prey, predator).
    Shape: (n_P + n_Z, n_Z).  Self-predation zeroed out.

    Parameters:
        theta_opt : optimal predator:prey ESD ratio  (default 10.0, Ward 2012)
        sigma_log : width in log10 space              (default 0.25, narrowed
                    from Ward's 0.5 to aid coexistence with few size classes)
    """
    prey_esd = np.concatenate([phyto_esd, zoo_esd])
    n_P = len(phyto_esd)
    n_Z = len(zoo_esd)
    log_ratio = np.log10(zoo_esd[None, :] / prey_esd[:, None])
    log_theta = np.log10(theta_opt)
    phiPZ = np.exp(-((log_ratio - log_theta) ** 2) / (2 * sigma_log ** 2))
    for j in range(n_Z):
        phiPZ[n_P + j, j] = 0.0
    return phiPZ


# --- Fish feeding weights ---------------------------------------------------

def _ramp(esd, esd_min, preference):
    """Linear ramp from 0 at esd_min to preference at max(esd)."""
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
    """Fish feeding weights on phytoplankton."""
    return _ramp(phyto_esd, p_esd_min, p_preference)


def compute_fish_weights_Z(zoo_esd, z_esd_min=30.0, z_preference=1.0):
    """Fish feeding weights on zooplankton."""
    return _ramp(zoo_esd, z_esd_min, z_preference)


# =============================================================================
# SIZE CLASSES
# =============================================================================
n_classes = 7
phyto_esd = generate_size_classes(n_classes, esd_min=0.5, esd_max=200)
zoo_esd   = generate_size_classes(n_classes, esd_min=5, esd_max=2000)

# =============================================================================
# INITIAL CONDITIONS
# =============================================================================
phyto_init = np.full(n_classes, 0.1)    # mmol N m-3 per size class
zoo_init   = np.full(n_classes, 0.01)   # mmol N m-3 per size class
N_init     = 0.01                       # mmol N m-3

# =============================================================================
# PHYTOPLANKTON PARAMETERS
# =============================================================================
mu_max      = compute_mu_max_maranon(phyto_esd)
K_s         = compute_K_s(phyto_esd)
m_P         = 0.1 * mu_max   # Banas (2011): linear mortality = 10% of mu_max
m_P_recycled = 1.0            # fraction recycled to N

# =============================================================================
# ZOOPLANKTON PARAMETERS
# =============================================================================
I_max = compute_I_max(zoo_esd)
gge   = compute_gge(zoo_esd)
m_Z   = 0.1    # quadratic mortality rate  [(mmol N m-3)^-1 d-1]
KsZ   = 3.0    # Type III half-saturation   [mmol N m-3]

# =============================================================================
# GRAZING KERNEL
# =============================================================================
phiPZ = compute_grazing_kernel(phyto_esd, zoo_esd)

# =============================================================================
# FISH PARAMETERS
# =============================================================================
w_P = compute_fish_weights_P(phyto_esd, p_esd_min=100.0, p_preference=0.3)
w_Z = compute_fish_weights_Z(zoo_esd, z_esd_min=30.0, z_preference=1.0)
fish_biomass = 1.0      # prescribed fish biomass (forcing)
fish_rate    = 0.005    # fish predation rate [d-1]

# =============================================================================
# CARIACO NUTRIENT SUPPLY (from time-series analysis)
# =============================================================================
N0_cariaco       = 5.5564    # mean sub-euphotic NO3, 50-70 m  [mmol N m-3]
dilution_cariaco = 0.016786  # exchange rate  [d-1]

# =============================================================================
# BUILD XSO MODEL
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
    # State variables
    'Nutrient':      {'value_label': 'N', 'value_init': N_init},
    'Phytoplankton': {'biomass_label': 'P', 'biomass_init': phyto_init,
                      'phyto_index': phyto_esd.tolist()},
    'Zooplankton':   {'biomass_label': 'Z', 'biomass_init': zoo_init,
                      'zoo_index': zoo_esd.tolist()},

    # Nutrient supply (Cariaco forcing)
    'N0':     {'forcing_label': 'N0', 'value': N0_cariaco},
    'Inflow': {'forcing': 'N0', 'rate': dilution_cariaco, 'var': 'N'},

    # Phytoplankton growth
    'Growth': {'resource': 'N', 'consumer': 'P',
               'halfsat': K_s, 'mu_max': mu_max},

    # Grazing and GGE
    'Grazing': {'resource': 'P', 'consumer': 'Z',
                'phiPZ': phiPZ, 'Imax': I_max, 'KsZ': KsZ},
    'GGE':     {'grazed_phyto': 'P', 'grazed_zoo': 'Z',
                'assimilated_consumer': 'Z', 'recycled_nutrient': 'N',
                'gge': gge},

    # Mortality
    'PhytoMortality': {'population': 'P', 'nutrient': 'N',
                       'rate': m_P, 'recycle_frac': m_P_recycled},
    'ZooMortality':   {'population': 'Z', 'rate': m_Z},

    # Fish
    'FishForcing': {'forcing_label': 'F_forcing', 'value': fish_biomass},
    'FishGrazing': {'phyto': 'P', 'zoo': 'Z',
                    'fish_forcing': 'F_forcing',
                    'w_P': w_P, 'w_Z': w_Z, 'rate': fish_rate},
}