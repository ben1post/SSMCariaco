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
    FishGrazing_Kernel,
)

# Imports
from npdxzxf_comps_detritus import (
    Detritus,
    GGE_Full_SizeDep_withD,
    PhytoMortality_toD_toN,
    ZooQuadraticMortality_toD,
    DetritusRemineralization,
    DetritusSinking,
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


def compute_fish_kernel_lognormal(esd, D_pref, sigma_log10):
    """Symmetric log-normal feeding kernel in ESD space, normalized to peak=1.

        K(D) = exp[ -(log10(D / D_pref))^2 / (2 * sigma_log10^2) ]

    Parameters
    ----------
    esd : array
        Equivalent spherical diameters of the prey size classes (µm).
    D_pref : float
        Preferred prey ESD where the kernel peaks (µm). For sardines,
        somewhere in the meso-zooplankton range (a few hundred µm) is a
        reasonable starting point.
    sigma_log10 : float
        Kernel width in log10(ESD) units. Larger = broader, more
        generalist filter-feeding. ~0.5–0.8 gives a kernel that spans
        roughly 1–2 decades of prey ESD on each side of D_pref, which
        is consistent with sardine biology (Andrades 2012; van der
        Lingen 1994).
    """
    log_ratio = np.log10(esd / D_pref)
    return np.exp(-log_ratio ** 2 / (2 * sigma_log10 ** 2))


import numpy as np


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
gge   = compute_gge(zoo_esd)
m_Z   = 0.1
KsZ   = 3.0

phiPZ = compute_grazing_kernel(phyto_esd, zoo_esd)


fish_biomass = 1.0
fish_rate    = 0.005
# linear:
#w_P = compute_fish_weights_P(phyto_esd)
#w_Z = compute_fish_weights_Z(zoo_esd)
# lognormal:
#D_pref_fish      = 500.0   # µm, preferred sardine prey size
#sigma_log10_fish = 0.6     # broad kernel, ~1 decade on each side
#kernel_P_fish    = compute_fish_kernel_lognormal(phyto_esd, D_pref_fish, sigma_log10_fish)
#kernel_Z_fish    = compute_fish_kernel_lognormal(zoo_esd,   D_pref_fish, sigma_log10_fish)
# data-based kernel:
kernel_P_fish,kernel_Z_fish = compute_fish_kernel_vdl_joint(phyto_esd,zoo_esd)


N0_cariaco       = 5.5564
dilution_cariaco = 0.016786


# Parameters for Detritus
D_init = 0.01                  # mmol N m-3
f_egest_D_zoo = 0.75           # Fasham: 75% egesta to D, 25% sloppy to N
f_mort_D_phyto = 0.9           # Fasham-style: most mortality to D
f_mort_D_zoo = 0.5             # Stock-style: half to D, half exported
k_remin = 0.1                  # d-1, warm tropical
d_e = 50.0                     # m, surface box depth
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
    'N0':             ConstantExternalNutrient,
    'Inflow':         LinearForcingInput,
    'Growth':         MonodGrowth_SizeBased,
    'Grazing':        SizebasedGrazingMatrix_Full_TypeIII,
    'GGE':            GGE_Full_SizeDep_withD,
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
                      'phyto_index': phyto_esd.tolist()},
    'Zooplankton':   {'biomass_label': 'Z', 'biomass_init': zoo_init,
                      'zoo_index': zoo_esd.tolist()},
    'Detritus': {'value_label': 'D', 'value_init': D_init},
    'N0':     {'forcing_label': 'N0', 'value': N0_cariaco},
    'Inflow': {'forcing': 'N0', 'rate': dilution_cariaco, 'var': 'N'},
    'Growth': {'resource': 'N', 'consumer': 'P',
               'halfsat': K_s, 'mu_max': mu_max},
    'Grazing': {'resource': 'P', 'consumer': 'Z',
                'phiPZ': phiPZ, 'Imax': I_max, 'KsZ': KsZ},
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
    input_vars=input_vars,
)