"""
CARIACO Observation Loader
==========================
Loads monthly observation data produced by the R export pipeline
(data/scenario_analysis/export_to_csv notebook) and returns model-ready
target vectors, labels, and bin definitions for model-observation comparison.
"""

import os
import numpy as np
import pandas as pd


# =============================================================================
# DEFAULT DATA PATH
# =============================================================================
DEFAULT_CSV_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "data", "processed", "cariaco_monthly_euphotic_dynamic.csv",
)


# =============================================================================
# TARGET BIN DEFINITIONS
# =============================================================================
# Single source of truth for which observation variables feed into the
# model-data comparison, in what order, and how the model should aggregate
# size-spectrum output onto each target.
#
# Each entry specifies:
#   - label:     human-readable name (used in plots & summary tables)
#   - column:    column name in the monthly CSV
#   - type:      'phyto' | 'zoo' | 'nutrient' (extensible: 'detritus', 'flux', ...)
#   - size_min, size_max: ESD bounds in µm (only for 'phyto' / 'zoo')
#
# To add new targets (e.g. PON, export flux) later, just append entries here.
TARGET_BIN_DEFINITIONS = [
    {'label': 'Pico (<2 µm)',   'column': 'pico_mmolN',      'type': 'phyto',
     'size_min': 0.0,   'size_max': 2.0},
    {'label': 'Nano (2-20 µm)', 'column': 'nano_mmolN',      'type': 'phyto',
     'size_min': 2.0,   'size_max': 20.0},
    {'label': 'Micro (>20 µm)', 'column': 'micro_mmolN',     'type': 'phyto',
     'size_min': 20.0,  'size_max': np.inf},
    {'label': 'Zoo >200 µm',    'column': 'zoo_gt200_mmolN', 'type': 'zoo',
     'size_min': 200.0, 'size_max': np.inf},
    {'label': 'Zoo >500 µm',    'column': 'zoo_gt500_mmolN', 'type': 'zoo',
     'size_min': 500.0, 'size_max': np.inf},
    {'label': 'NO3',            'column': 'NO3_mmolN',       'type': 'nutrient'},
]


# =============================================================================
# LOADER
# =============================================================================
def load_cariaco_targets(regime='all', csv_path=DEFAULT_CSV_PATH,
                         bin_definitions=TARGET_BIN_DEFINITIONS):
    """
    Load CARIACO monthly observations and build the target vector for
    model-data comparison.

    Parameters
    ----------
    regime : str
        One of:
          - 'all'       : average across all months (ignores upwelling class)
          - 'upwelling' : only months classified as 'upwelling'
          - 'relaxed'   : only months classified as 'relaxed'
          - 'strong' | 'moderate' | 'weak' : filter by detailed ui column
    csv_path : str
        Path to the monthly CSV produced by the R export pipeline.
    bin_definitions : list of dict
        Target bin definitions. Defaults to TARGET_BIN_DEFINITIONS.

    Returns
    -------
    obs_vec : np.ndarray, shape (n_targets,)
        Mean of each target variable across the (filtered) months.
    labels : list of str
        Human-readable target labels, same order as obs_vec.
    bin_definitions : list of dict
        The bin definitions used, same order as obs_vec.
    monthly_df : pd.DataFrame
        The filtered monthly dataframe with only the target columns
        (plus 'date', 'time_month', 'upwelling', 'ui' for reference).
        Useful for boxplots / variance analysis.
    """
    df = pd.read_csv(csv_path)

    # Filter by regime
    if regime == 'all':
        filtered = df
    elif regime in ('upwelling', 'relaxed'):
        filtered = df[df['upwelling'] == regime]
    elif regime in ('strong', 'moderate', 'weak'):
        filtered = df[df['ui'] == regime]
    else:
        raise ValueError(
            f"Unknown regime '{regime}'. Expected one of: "
            f"'all', 'upwelling', 'relaxed', 'strong', 'moderate', 'weak'."
        )

    # Extract target columns + context columns for the returned monthly df
    target_cols = [b['column'] for b in bin_definitions]
    context_cols = [c for c in ('date', 'time_month', 'upwelling', 'ui')
                    if c in filtered.columns]
    monthly_df = filtered[context_cols + target_cols].copy()

    # Build obs vector (mean across months, NaN-safe)
    obs_vec = np.array([monthly_df[col].mean(skipna=True) for col in target_cols])

    # Fail loud if any target came back all-NaN — cost function can't handle it
    for label, val in zip([b['label'] for b in bin_definitions], obs_vec):
        if np.isnan(val):
            raise ValueError(
                f"Target '{label}' has no valid observations in regime '{regime}'."
            )

    labels = [b['label'] for b in bin_definitions]

    return obs_vec, labels, bin_definitions, monthly_df