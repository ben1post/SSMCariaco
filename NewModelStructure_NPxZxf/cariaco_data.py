"""
CARIACO Observation Data
========================
Loads time-series data, converts to model units (mmol N m-3),
and provides aggregation utilities for model-observation comparison.
"""

import numpy as np
import pandas as pd

# =============================================================================
# FILE PATHS (adjust as needed)
# =============================================================================
ENVDATA_PATH = "../DATA/processed/CARIACO_EnvData_combined.csv"
NUTRIENT_PATH = "NO3_euphotic_dynamic.csv"

# =============================================================================
# UNIT CONVERSION CONSTANTS
# =============================================================================
EUPHOTIC_DEPTH = 44.92    # m
C_TO_CHL      = 50.0     # mg C : mg Chl
C_TO_DW       = 0.4      # mg C : mg Dry Weight (zooplankton)
REDFIELD_N_C  = 16 / 106 # mmol N : mmol C
MW_CARBON     = 12.01    # g mol-1


# =============================================================================
# LOAD OBSERVATIONS
# =============================================================================

def load_phyto_obs():
    """Load phytoplankton size-fractionated data (mmol N m-3)."""
    raw = pd.read_csv(ENVDATA_PATH)
    cols = ['pico_abs', 'nano_abs', 'micro_abs']
    df = raw[cols].dropna().copy()
    for col in cols:
        df[col] = (df[col] / EUPHOTIC_DEPTH) * C_TO_CHL / MW_CARBON * REDFIELD_N_C
    df.columns = ['Pico (<2 µm)', 'Nano (2-20 µm)', 'Micro (>20 µm)']
    return df


def load_zoo_obs():
    """Load zooplankton cumulative net data (mmol N m-3)."""
    raw = pd.read_csv(ENVDATA_PATH)
    df = pd.DataFrame()
    df['>200 µm Net'] = raw['BIOMASS_200']
    df['>500 µm Net'] = raw['BIOMASS_500']
    df = df.dropna().copy()
    for col in df.columns:
        df[col] = (df[col] * C_TO_DW) / MW_CARBON * REDFIELD_N_C
    return df


def load_nutrient_obs():
    """Load euphotic-zone NO3 observations (already mmol m-3)."""
    df = pd.read_csv(NUTRIENT_PATH)
    df = df[['NO3_euphotic']].dropna().copy()
    df.columns = [f'NO3 (0-{int(EUPHOTIC_DEPTH)}m)']
    return df


# =============================================================================
# FRACTIONAL OVERLAP UTILITIES
# =============================================================================

def get_log_bin_edges(centers):
    """Calculate N+1 bin edges for N log-spaced centers."""
    q = centers[1] / centers[0]
    half_step = np.sqrt(q)
    edges = np.zeros(len(centers) + 1)
    edges[0] = centers[0] / half_step
    edges[1:] = centers * half_step
    return edges


def get_fraction_in_range(lower, upper, target_min, target_max):
    """Fractional overlap of log-bin [lower, upper] with [target_min, target_max]."""
    overlap_min = max(lower, target_min)
    overlap_max = min(upper, target_max)
    if overlap_min >= overlap_max:
        return 0.0
    return ((np.log10(overlap_max) - np.log10(overlap_min))
            / (np.log10(upper) - np.log10(lower)))


def aggregate_model_state(ss_phyto, ss_zoo, ss_nut, p_esd, z_esd):
    """
    Aggregate model size-spectrum output into observation categories.

    Returns: (model_phyto, model_zoo, model_nut)
        model_phyto: [micro, nano, pico]  (mmol N m-3)
        model_zoo:   [>200 µm, >500 µm]   (mmol N m-3, cumulative)
        model_nut:   [NO3]                 (mmol N m-3)
    """
    # --- Phytoplankton: pico / nano / micro ---
    p_edges = get_log_bin_edges(p_esd)
    micro, nano, pico = 0.0, 0.0, 0.0
    for i in range(len(p_esd)):
        lo, hi = p_edges[i], p_edges[i + 1]
        b = ss_phyto[i]
        pico  += b * get_fraction_in_range(lo, hi, 1e-9, 2.0)
        nano  += b * get_fraction_in_range(lo, hi, 2.0, 20.0)
        micro += b * get_fraction_in_range(lo, hi, 20.0, 1e9)
    model_phyto = [micro, nano, pico]

    # --- Zooplankton: cumulative nets ---
    z_edges = get_log_bin_edges(z_esd)
    zoo_gt200, zoo_gt500 = 0.0, 0.0
    for i in range(len(z_esd)):
        lo, hi = z_edges[i], z_edges[i + 1]
        b = ss_zoo[i]
        zoo_gt200 += b * get_fraction_in_range(lo, hi, 200.0, 1e9)
        zoo_gt500 += b * get_fraction_in_range(lo, hi, 500.0, 1e9)
    model_zoo = [zoo_gt200, zoo_gt500]

    return model_phyto, model_zoo, [float(ss_nut)]