"""
cariaco_data_processor.py
=========================
Processes CARIACO time-series data to generate steady-state forcing parameters
and verification baseline data in model units (mmol N m-3).
"""

import pandas as pd
import numpy as np

# =============================================================================
# 1. CONSTANTS & CONVERSION FACTORS
# =============================================================================

# Biogeochemical assumptions
C_TO_CHL = 50.0          # mg C : mg Chl a
C_TO_DW = 0.4            # mg C : mg Dry Weight (zooplankton ~40% C)
REDFIELD_N_C = 16 / 106  # mmol N : mmol C
ATOMIC_WEIGHT_C = 12.01  # mg / mmol

# System parameters
EUPHOTIC_DEPTH = 44.92   # m (grand mean from CARIACO combined data)

# File paths (Adjust as needed)
FILE_COMBINED = "../DATA/processed/CARIACO_EnvData_combined.csv"
FILE_NO3_DYN = "DataAnalysis/NO3_euphotic_dynamic.csv"

# =============================================================================
# 2. DERIVE MODEL FORCINGS (Ported from R logic)
# =============================================================================

def calculate_steady_state_forcings(mean_temp=24.34, mean_pp_areal=1202.97):
    """
    Calculates physical and biological forcings based on mean CARIACO conditions.
    Using hardcoded means from the R script output for stability, but this can 
    be wired to dynamic dataframe inputs if needed.
    """
    # 1. f-ratio (Laws et al. 2011)
    f_ratio = (0.5857 - 0.0165 * mean_temp) * mean_pp_areal / (51.7 + mean_pp_areal)
    
    # 2. New Production (mg C m-2 d-1)
    new_prod_area = f_ratio * mean_pp_areal
    
    # 3. Nutrient Flux (mmol N m-3 d-1) - This is your new Linear Input for N
    # Converts: (mg C m-2 d-1) / m / (mg/mmol C) * (mmol N / mmol C)
    new_nutrient_flux = (new_prod_area / EUPHOTIC_DEPTH) / ATOMIC_WEIGHT_C * REDFIELD_N_C
    
    return {
        'temperature_C': mean_temp,
        'f_ratio': f_ratio,
        'new_production_mgC_m2_d': new_prod_area,
        'linear_N_input_flux': new_nutrient_flux
    }

# =============================================================================
# 3. PROCESS VERIFICATION DATA (Model Units: mmol N m-3)
# =============================================================================

def process_observational_data():
    """
    Loads raw CARIACO data, applies unit conversions, and returns summary stats 
    for model verification.
    """
    # Load raw data
    try:
        cariaco_data = pd.read_csv(FILE_COMBINED)
        nut_data = pd.read_csv(FILE_NO3_DYN)
    except FileNotFoundError as e:
        print(f"Warning: Data file not found. {e}")
        return None, None
        
    # --- A. Phytoplankton (mg Chl m-2 -> mmol N m-3) ---
    phyto_cols = ['micro_abs', 'nano_abs', 'pico_abs']
    phyto_data = cariaco_data[phyto_cols].dropna().copy()
    
    for col in phyto_cols:
        # (Areal / Depth) * C:Chl / atomic_weight_C * Redfield
        phyto_data[col] = (phyto_data[col] / EUPHOTIC_DEPTH) * C_TO_CHL / ATOMIC_WEIGHT_C * REDFIELD_N_C
    
    phyto_data.columns = ['Phyto_Micro', 'Phyto_Nano', 'Phyto_Pico']
    
    # --- B. Zooplankton (mg DW m-3 -> mmol N m-3) ---
    zoo_data = pd.DataFrame()
    zoo_data['Zoo_gt_200'] = cariaco_data['BIOMASS_200']
    zoo_data['Zoo_gt_500'] = cariaco_data['BIOMASS_500']
    zoo_data = zoo_data.dropna().copy()
    
    for col in zoo_data.columns:
        # mg DW * C:DW / atomic_weight_C * Redfield
        zoo_data[col] = (zoo_data[col] * C_TO_DW) / ATOMIC_WEIGHT_C * REDFIELD_N_C
        
    # --- C. Nutrients (Already in µmol/L == mmol N m-3) ---
    n_data = nut_data[['NO3_euphotic']].dropna().copy()
    n_data.columns = ['NO3_Surface']
    
    # Combine into a single averages dictionary
    verification_means = {
        'Phyto_Micro_mean': phyto_data['Phyto_Micro'].mean(),
        'Phyto_Nano_mean': phyto_data['Phyto_Nano'].mean(),
        'Phyto_Pico_mean': phyto_data['Phyto_Pico'].mean(),
        'Zoo_gt_200_mean': zoo_data['Zoo_gt_200'].mean(),
        'Zoo_gt_500_mean': zoo_data['Zoo_gt_500'].mean(),
        'NO3_Surface_mean': n_data['NO3_Surface'].mean()
    }
    
    # Return both the raw converted dataframes (for boxplots) and the means
    return verification_means, (phyto_data, zoo_data, n_data)

# =============================================================================
# EXECUTE & PRINT SUMMARY
# =============================================================================

if __name__ == "__main__":
    forcings = calculate_steady_state_forcings()
    verif_means, raw_dfs = process_observational_data()
    
    print("=== MODEL FORCINGS ===")
    for k, v in forcings.items():
        print(f"{k}: {v:.6f}")
        
    if verif_means:
        print("\n=== VERIFICATION BASELINE (mmol N m-3) ===")
        for k, v in verif_means.items():
            print(f"{k}: {v:.5f}")