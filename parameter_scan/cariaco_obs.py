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
    {'label': 'PP',    'column': 'PP_mmolN_m3_d',   'type': 'pp'},
    {'label': 'Export',  'column': 'export_flux_corrected_mmolN','type': 'export'},
]

# Passthrough diagnostic columns — carried into monthly_df for inspection,
# NOT used as model-data comparison targets.
DIAGNOSTIC_COLS = [
    'Chl_niskin_mgm3',
    'Phaeo_niskin_mgm3',
    'Chl_niskin_mmolN',
    'PhaeoChl_ratio',
    'Temp_C',            # euphotic-zone mean temperature (mean over 0->depth_cutoff)
]

# =============================================================================
# LOADER
# =============================================================================
def load_cariaco_targets(regime='all', csv_path=DEFAULT_CSV_PATH,
                         bin_definitions=TARGET_BIN_DEFINITIONS,
                         regime_col=None, agg='mean', months='hplc'):
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
          - 'transition': near-threshold boundary months (regime_adjusted only)
          - 'strong' | 'moderate' | 'weak' : filter by detailed ui column
    regime_col : str or None
        Column used for the upwelling/relaxed/transition split. If None (default),
        uses 'regime_adjusted' when present (boundary-corrected), else 'upwelling'.
    agg : {'mean', 'median'}
        Statistic used to collapse each regime's months into the obs target vector
        AND the model forcing. 'median' is the more defensible choice for steady-state
        regime runs (the large-cell bloom tail is a real signal a mean over-weights and
        a median deliberately down-weights); 'mean' integrates the tail. Applied
        identically to obs_vec and to Inflow__FN / Inflow__de so target and forcing stay
        on the same statistic.
    months : {'hplc', 'available', 'complete'}
        Month set the regime statistic is computed over. 'hplc' (default) restricts to
        months where the phytoplankton size spectrum resolves, so the forcing (F_N, d_e)
        and the size targets describe the SAME months — the causally-consistent choice
        for the model-obs comparison (matters for upwelling, where all-months F_N is
        inflated relative to the HPLC-observed months). 'available' uses each column's
        own non-NaN months (max coverage, original behaviour; forcing and size targets
        may span different months). 'complete' requires every target column present
        (often empty — e.g. no upwelling month has HPLC + zoo + export together).
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
        The filtered monthly dataframe with the target columns, the per-month
        forcing columns ('FN_mmolN_m2_d', 'depth_cutoff'), and context columns
        ('date', 'time_month', 'upwelling', 'ui', ...). Useful for boxplots /
        variance analysis and the regime-forcing figure.
    forcing : dict
        Regime-specific model forcing keyed by XSO parameter name:
        {'Inflow__FN': float, 'Inflow__de': float}. Pass directly as
        `fixed_overrides` to run_xso_parscan.
    """
    df = pd.read_csv(csv_path)

    # Resolve which classification column defines the upwelling/relaxed split.
    # Prefer the boundary-corrected `regime_adjusted` (near-threshold relaxed months
    # adjacent to an upwelling event are routed to a 'transition' class and excluded
    # from BOTH composites; produced by depth_profile_data.r::get_full_scenario_data).
    # Fall back to the original sharp-threshold `upwelling` for CSVs predating it.
    if regime_col is None:
        regime_col = 'regime_adjusted' if 'regime_adjusted' in df.columns else 'upwelling'

    # Filter by regime
    if regime == 'all':
        filtered = df
    elif regime in ('upwelling', 'relaxed', 'transition'):
        filtered = df[df[regime_col] == regime]
    elif regime in ('strong', 'moderate', 'weak'):
        filtered = df[df['ui'] == regime]
    else:
        raise ValueError(
            f"Unknown regime '{regime}'. Expected one of: 'all', 'upwelling', "
            f"'relaxed', 'transition', 'strong', 'moderate', 'weak'."
        )

    target_cols = [b['column'] for b in bin_definitions]

    # Optionally restrict to a common month set so the forcing and the obs targets
    # describe the SAME months (causal consistency for the model-obs comparison):
    #   'available' : every column over all its non-NaN months in the regime (max
    #                 coverage; forcing and size targets may span different months)
    #   'hplc'      : only months where the phytoplankton size spectrum resolves
    #   'complete'  : only months where every target column is present (strictest)
    phyto_cols = [b['column'] for b in bin_definitions if b['type'] == 'phyto']
    if months == 'available':
        subset = filtered
    elif months == 'hplc':
        subset = filtered.dropna(subset=phyto_cols)
    elif months == 'complete':
        subset = filtered.dropna(subset=target_cols)
    else:
        raise ValueError(
            f"months must be 'available', 'hplc', or 'complete', got '{months}'."
        )

    # Extract target + context + forcing columns for the returned monthly df.
    # The per-month forcing columns (F_N and box depth) are carried so the
    # regime-forcing figure can boxplot their monthly distributions; they feed
    # the `forcing` dict below but are NOT model-data comparison targets.
    context_cols = [c for c in ('date', 'time_month', 'upwelling', 'ui',
                                'regime_adjusted', 'boundary_flag')
                    if c in subset.columns]
    forcing_cols = [c for c in ('FN_mmolN_m2_d', 'depth_cutoff')
                    if c in subset.columns]
    diagnostic_cols = [c for c in DIAGNOSTIC_COLS if c in subset.columns]
    keep_cols = list(dict.fromkeys(
        context_cols + forcing_cols + target_cols + diagnostic_cols))
    monthly_df = subset[keep_cols].copy()

    # Collapse each regime's months to one value per target with the chosen
    # statistic (NaN-safe). 'median' down-weights the large-cell bloom tail (the
    # steady-state-appropriate choice); 'mean' integrates it.
    if agg not in ('mean', 'median'):
        raise ValueError(f"agg must be 'mean' or 'median', got '{agg}'.")
    obs_vec = np.array([getattr(monthly_df[col], agg)(skipna=True)
                        for col in target_cols])

    # Fail loud if any target came back all-NaN — cost function can't handle it
    for label, val in zip([b['label'] for b in bin_definitions], obs_vec):
        if np.isnan(val):
            raise ValueError(
                f"Target '{label}' has no valid observations in regime '{regime}' "
                f"under months='{months}'. Pass a reduced bin_definitions (e.g. phyto "
                f"+ NO3 + PP) or months='available' if this target is genuinely sparse."
            )

    # Regime-specific forcing for the model (F_N, d_e), collapsed with the SAME
    # `agg` statistic as the obs targets so forcing and target stay consistent.
    # Both columns are computed upstream in the R pipeline at monthly resolution.
    forcing = {
        'Inflow__FN': float(getattr(subset['FN_mmolN_m2_d'], agg)(skipna=True)),
        'Inflow__de': float(getattr(subset['depth_cutoff'], agg)(skipna=True)),
    }
    for k, v in forcing.items():
        if np.isnan(v):
            raise ValueError(
                f"Cannot compute forcing '{k}' for regime '{regime}' — "
                f"source column has no valid values."
            )
    
    labels = [b['label'] for b in bin_definitions]

    return obs_vec, labels, bin_definitions, monthly_df, forcing