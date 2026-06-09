"""
baseline_r0_testing_setups.py — clean-room R0 ladder setups
===========================================================
Setups for the step-by-step R0 stability diagnosis (2026-06-08), built on
`baseline_r0_testing_comps.py`. Self-contained; no inherited baggage.

RUNG 1 (minimal omnivory baseline): N-P-Z size-structured, distributed Holling
Type III OMNIVORY, quadratic zoo closure, temperature (Cloern Q10), Stock
F_N/d_e supply + phyto sinking, allometric rates. Anchor values are the
NON-SUSPECT ends: K_sZ uniform 0.30 (top of literature range = weaker grazing),
kernel std 0.25 (the wider 2σ² convention). recycle_fraction default 0.0
(full export = Banas open-system convention).

Deferred-to-later-rung suspects: low K_sZ (0.15), narrow Mattern kernel
(σ=0.15 ÷σ² = std 0.106), recycle_fraction > 0, detritus pool.

SOLVER: RK45 + relaxed atol (the real speedup lever; NPZ oscillatory ≠ stiff,
2026-05-22). Setups carry only RK45/atol/rtol; pass a looser
`instability_neg_threshold` (or anything non-default) at scan / run_single_point
call time — do NOT bake custom solver params into setups.

Grid size via env var MS3_R0_N (default 40); run 20 / 80 for resolution checks.
"""

import os
import numpy as np
import xso

from baseline_r0_testing_comps import (
    Nutrient, PhytoSizeSpectrum, ZooSizeSpectrum,
    StockNutrientSupply, ConstantTemperatureForcing,
    MonodGrowth_T, compute_grazing_kernel, DistributedGrazing_TypeIII_T,
    DistributedGrazingRouter_split,
    PhytoLinearLoss_split, ZooQuadraticLoss_split, PhytoSinking_export,
)

# ---- grid -------------------------------------------------------------------
N_CLASSES       = int(os.environ.get('MS3_R0_N', 40))
PHYTO_ESD_MIN   = 0.2      # µm (Sieburth Pico floor)
PHYTO_ESD_MAX   = 200.0    # µm
ZOO_PHYTO_RATIO = 10.0     # r = 10 (Survey §9)
phyto_esd = np.logspace(np.log10(PHYTO_ESD_MIN), np.log10(PHYTO_ESD_MAX), N_CLASSES)
zoo_esd   = ZOO_PHYTO_RATIO * phyto_esd

# ---- allometries: SINGLE SOURCE = Banas 2011 Table 2 (verbatim) -------------
# Hard rule (2026-06-08): every allometry from ONE source, prefactor+exponent
# together, no splicing. Banas 2011 Table 2 gives all three on the ESD axis.
mu_max_arr = 2.6  * phyto_esd ** (-0.45)   # Banas 2011 Table 2 (←Tang 1995)
ks_arr     = 0.1  * phyto_esd ** ( 1.0)    # Banas 2011 Table 2 (Eppley 1969 surface-area)
Imax_arr   = 26.0 * zoo_esd   ** (-0.40)   # Banas 2011 Table 2
mP_arr     = 0.1  * mu_max_arr             # m_P = 0.1·μ_max (Banas 2011)

# ---- scalar params (RUNG-1 anchors) -----------------------------------------
GAMMA_VAL          = 0.25     # Γ gross growth efficiency (Stock 2008)
M_Z_VAL            = 0.30     # quadratic closure coeff (Banas Eq.10 ≈0.26)
KSZ_ANCHOR         = 0.30     # uniform grazing half-sat [mmol N m-3] (non-suspect end)
SIGMA_ANCHOR       = 0.25     # kernel std, 2σ² convention (wider, non-suspect)
THETA_OPT          = 10.0     # predator:prey ESD ratio
RECYCLE_FRAC_DEF   = 0.0      # loss fate: 0 = full export (Banas open); 1 = full recycle

Q10_GROW  = 1.62             # Cloern 2018 growth Q10
Q10_GRAZE = 2.48             # Cloern 2018 grazing Q10
T_REF     = 20.0             # °C
T_DEFAULT = 24.0             # °C placeholder (override per regime)

FN_DEFAULT = 2.67            # mmol N m-2 d-1 ('all'-regime mean)
DE_DEFAULT = 50.0            # m
W_SINK     = 5.0             # m/d (Stock/MS3-as-built)

N_INIT = 1.0; P_INIT = 1e-3; Z_INIT = 1e-3
phyto_init = np.full(N_CLASSES, P_INIT)
zoo_init   = np.full(N_CLASSES, Z_INIT)

# ---- omnivory kernel (rung-1 anchor width; 2σ² convention) ------------------
phiPZ_omni = compute_grazing_kernel(phyto_esd, zoo_esd, mode='omni',
                                    theta_opt=THETA_OPT, sigma_log=SIGMA_ANCHOR,
                                    convention='2sigma2')

# ---- solver / time ----------------------------------------------------------
# RK45 + relaxed atol only. Pass instability_neg_threshold etc. at call time.
IVP_SOLVER_KWARGS = {'method': 'RK45', 'atol': 1e-6, 'rtol': 1e-4}
IVP_TIME_END   = 5000.0
ivp_time_array = np.arange(0.0, IVP_TIME_END, 1.0)
STAB_TIME = [0.0, 1.0]
SLIM_OUTPUT_VARS = {'Nutrient__value', 'Phytoplankton__biomass', 'Zooplankton__biomass'}

# ---- model ------------------------------------------------------------------
model_r0 = xso.create({
    'Nutrient':      Nutrient,
    'Phytoplankton': PhytoSizeSpectrum,
    'Zooplankton':   ZooSizeSpectrum,
    'Inflow':        StockNutrientSupply,
    'Temperature':   ConstantTemperatureForcing,
    'Growth':        MonodGrowth_T,
    'Grazing':       DistributedGrazing_TypeIII_T,
    'GrazingRouter': DistributedGrazingRouter_split,
    'PhytoLoss':     PhytoLinearLoss_split,
    'ZooLoss':       ZooQuadraticLoss_split,
    'PhytoSinking':  PhytoSinking_export,
})


def make_r0_input_vars(recycle_fraction=RECYCLE_FRAC_DEF, phiPZ=phiPZ_omni,
                       ksz=KSZ_ANCHOR):
    """Rung-1 input_vars. `recycle_fraction` sets the loss fate on ALL three loss
    terms (grazing sloppy feeding, phyto mortality, zoo closure) identically."""
    return {
        'Nutrient':      {'value_label': 'N', 'value_init': N_INIT},
        'Phytoplankton': {'biomass_label': 'P', 'biomass_init': phyto_init,
                          'phyto_esd_index': phyto_esd.tolist(),
                          'phyto_esd_label': 'phyto_esd'},
        'Zooplankton':   {'biomass_label': 'Z', 'biomass_init': zoo_init,
                          'zoo_esd_index': zoo_esd.tolist(),
                          'zoo_esd_label': 'zoo_esd'},
        'Inflow':        {'var': 'N', 'FN': FN_DEFAULT,
                          'de': DE_DEFAULT, 'de_label': 'de'},
        'Temperature':   {'forcing_label': 'temperature', 'value': T_DEFAULT},
        'Growth':        {'resource': 'N', 'consumer': 'P', 'temperature': 'temperature',
                          'mu_max': mu_max_arr, 'halfsat': ks_arr,
                          'q10': Q10_GROW, 't_ref': T_REF},
        'Grazing':       {'resource': 'P', 'consumer': 'Z', 'temperature': 'temperature',
                          'phiPZ': phiPZ, 'Imax': Imax_arr,
                          'KsZ': np.full(N_CLASSES, ksz),
                          'q10': Q10_GRAZE, 't_ref': T_REF},
        'GrazingRouter': {'grazed_phyto': 'P', 'grazed_zoo': 'Z',
                          'assimilated_consumer': 'Z', 'recycled_nutrient': 'N',
                          'gamma': GAMMA_VAL, 'recycle_fraction': recycle_fraction},
        'PhytoLoss':     {'population': 'P', 'recycled_nutrient': 'N',
                          'rate': mP_arr, 'recycle_fraction': recycle_fraction},
        'ZooLoss':       {'population': 'Z', 'recycled_nutrient': 'N',
                          'rate': M_Z_VAL, 'recycle_fraction': recycle_fraction},
        'PhytoSinking':  {'population': 'P', 'w_sink': W_SINK, 'de': 'de'},
    }


model_setup_r0_slim = xso.setup(
    solver='solve_ivp', model=model_r0, time=ivp_time_array,
    input_vars=make_r0_input_vars(), output_vars=SLIM_OUTPUT_VARS,
    solver_kwargs=IVP_SOLVER_KWARGS)

model_setup_r0_stability = xso.setup(
    solver='stability', model=model_r0, time=STAB_TIME,
    input_vars=make_r0_input_vars())
