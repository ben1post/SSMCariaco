"""
Allometric Parameter Generator
===============================
Individual functions for each size-dependent parameter.
All scaling relationships follow Banas et al. (2011).
"""

import numpy as np


def generate_size_classes(n, esd_min, esd_max):
    """Generate log-spaced size classes (µm ESD)."""
    return np.logspace(np.log10(esd_min), np.log10(esd_max), n)


def _volume(esd):
    """ESD (µm) → spherical biovolume."""
    return (np.pi / 6) * esd ** 3


# --- Phytoplankton ---------------------------------------------------------

def compute_mu_max(esd, mu_max_ref, esd_ref):
    """Maximum growth rate: µ_max = ref * (V/V_ref)^(-1/4)."""
    return mu_max_ref * (_volume(esd) / _volume(esd_ref)) ** (-1 / 4)


def compute_K_s(esd, ks_ref, esd_ref):
    """Half-saturation constant: K_s = ref * (V/V_ref)^(1/4)."""
    return ks_ref * (_volume(esd) / _volume(esd_ref)) ** (1 / 4)


# --- Zooplankton -----------------------------------------------------------

def compute_I_max(esd, imax_ref, esd_ref):
    """Maximum ingestion rate: I_max = ref * (V/V_ref)^(-0.16)."""
    return imax_ref * (_volume(esd) / _volume(esd_ref)) ** (-0.16)


def compute_gge(esd, gge_small, gge_large):
    """GGE: linear interpolation in log-ESD space from small to large."""
    log_esd = np.log10(esd)
    if len(esd) > 1:
        frac = (log_esd - log_esd[0]) / (log_esd[-1] - log_esd[0])
    else:
        frac = np.array([0.5])
    return gge_small + (gge_large - gge_small) * frac


# --- Grazing kernel --------------------------------------------------------

def compute_grazing_kernel(phyto_esd, zoo_esd, theta_opt, sigma_log):
    """Log-normal grazing preference matrix φ(prey, predator).

    Returns shape (n_P + n_Z, n_Z). Self-predation zeroed out.
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


# --- Fish feeding weights --------------------------------------------------

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


def compute_fish_weights_P(phyto_esd, p_esd_min, p_preference):
    """Fish feeding weights on phytoplankton."""
    return _ramp(phyto_esd, p_esd_min, p_preference)


def compute_fish_weights_Z(zoo_esd, z_esd_min, z_preference):
    """Fish feeding weights on zooplankton."""
    return _ramp(zoo_esd, z_esd_min, z_preference)