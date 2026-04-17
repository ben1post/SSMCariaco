# npxzxf_models.py

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import xso

from npxzxf_comps import (
    Nutrient, PhytoSizeSpectrum, ZooSizeSpectrum,
    ConstantExternalNutrient, ConstantFishForcing,
    LinearForcingInput, MonodGrowth_SizeBased,
    SizebasedGrazingMatrix_Full_TypeIII, GGE_Full_SizeDep,
    PhytoMortality_toN, ZooQuadraticMortality,
    FishGrazing_SizeBased,ChemostatInput,
)

from allometric_params import (
    generate_size_classes,
    compute_mu_max, compute_K_s,
    compute_I_max, compute_gge,
    compute_grazing_kernel,
    compute_fish_weights_P, compute_fish_weights_Z,
)


# =============================================================================
# SIZE CLASSES
# =============================================================================

n_classes = 7
phyto_esd = generate_size_classes(n_classes, 0.5, 200)
zoo_esd   = generate_size_classes(n_classes, 5, 2000)

# =============================================================================
# INITIAL CONDITIONS
# =============================================================================

phyto_init = np.full(n_classes, 0.1)
zoo_init   = np.full(n_classes, 0.01)

# =============================================================================
# PHYTOPLANKTON PARAMETERS
# =============================================================================

mu_max = compute_mu_max(phyto_esd, mu_max_ref=2.0, esd_ref=1.0)
K_s    = compute_K_s(phyto_esd, ks_ref=0.1, esd_ref=1.0)
m_P    = np.full(n_classes, 0.01)

# =============================================================================
# ZOOPLANKTON PARAMETERS
# =============================================================================

I_max = compute_I_max(zoo_esd, imax_ref=20.0, esd_ref=10.0)
gge   = compute_gge(zoo_esd, gge_small=0.35, gge_large=0.15)
m_Z   = 0.01

# =============================================================================
# GRAZING KERNEL
# =============================================================================

phiPZ = compute_grazing_kernel(phyto_esd, zoo_esd, theta_opt=10.0, sigma_log=0.15)

# =============================================================================
# FISH PARAMETERS
# =============================================================================

w_P = compute_fish_weights_P(phyto_esd, p_esd_min=100.0, p_preference=0.3)
w_Z = compute_fish_weights_Z(zoo_esd, z_esd_min=30.0, z_preference=1.0)

# =============================================================================
# NUTRIENT SUPPLY
# =============================================================================

nutrient_input = 0.01
dilution_rate  = 1.0


# =============================================================================
# BASE INPUT VARS
# =============================================================================

BASE_INPUT_VARS = {
    # State variables
    'Nutrient': {'value_label': 'N', 'value_init': .01},
    'Phytoplankton': {'biomass_label': 'P', 'biomass_init': phyto_init,
                      'phyto_index': phyto_esd.tolist()},
    'Zooplankton': {'biomass_label': 'Z', 'biomass_init': zoo_init,
                    'zoo_index': zoo_esd.tolist()},
    # Nutrient supply
    'N0': {'forcing_label': 'N0', 'value': nutrient_input},
    'Inflow': {'forcing': 'N0', 'rate': dilution_rate, 'var': 'N'},
    # Growth
    'Growth': {'resource': 'N', 'consumer': 'P',
               'halfsat': K_s, 'mu_max': mu_max},
    # Grazing and GGE
    'Grazing': {'resource': 'P', 'consumer': 'Z',
                'phiPZ': phiPZ, 'Imax': I_max, 'KsZ': 3.0},
    'GGE': {'grazed_phyto': 'P', 'grazed_zoo': 'Z',
            'assimilated_consumer': 'Z', 'recycled_nutrient': 'N',
            'gge': gge},
    # Mortality
    'PhytoMortality': {'population': 'P', 'nutrient': 'N', 'rate': m_P},
    'ZooMortality': {'population': 'Z', 'rate': m_Z},
    # Fish forcing
    'FishForcing': {'forcing_label': 'F_forcing', 'value': 1.0},
    'FishGrazing': {'phyto': 'P', 'zoo': 'Z',
                    'fish_forcing': 'F_forcing',
                    'w_P': w_P, 'w_Z': w_Z, 'rate': 0.05},
}


# =============================================================================
# MODEL OBJECT
# =============================================================================

model = xso.create({
    'Nutrient': Nutrient,
    'Phytoplankton': PhytoSizeSpectrum,
    'Zooplankton': ZooSizeSpectrum,
    'N0': ConstantExternalNutrient,
    'Inflow': LinearForcingInput,
    'Growth': MonodGrowth_SizeBased,
    'Grazing': SizebasedGrazingMatrix_Full_TypeIII,
    'GGE': GGE_Full_SizeDep,
    'PhytoMortality': PhytoMortality_toN,
    'ZooMortality': ZooQuadraticMortality,
    'FishForcing': ConstantFishForcing,
    'FishGrazing': FishGrazing_SizeBased,
})


# =============================================================================
# MODEL SETUPS
# =============================================================================

model_setup_ivp = xso.setup(
    solver='solve_ivp',
    model=model,
    time=np.arange(0, 5000),
    input_vars=BASE_INPUT_VARS,
)

model_setup_ivp_1k = xso.setup(
    solver='solve_ivp',
    model=model,
    time=np.arange(0, 1000),
    input_vars=BASE_INPUT_VARS,
)

model_setup_stability = xso.setup(
    solver='stability',
    model=model,
    time=[0, 1],
    input_vars=BASE_INPUT_VARS,
)


# =============================================================================
# CARIACO-SPECIFIC SETUP (for parameter scans)
# =============================================================================

CARIACO_INPUT_VARS = BASE_INPUT_VARS.copy()
CARIACO_INPUT_VARS.update({
    'N0':      {**BASE_INPUT_VARS['N0'],      'value': 5.5564},
    'Inflow':  {**BASE_INPUT_VARS['Inflow'],  'rate': 0.016786},
    # 'Nutrient': {**BASE_INPUT_VARS['Nutrient'], 'value_init': 2.0158},
})

model_setup_ivp_cariaco = xso.setup(
    solver='solve_ivp',
    model=model,
    time=np.arange(0, 2000),
    input_vars=CARIACO_INPUT_VARS,
)

# =============================================================================
# RUN FUNCTIONS
# =============================================================================

def run_model_test(i=None, time=None):
    """Run the model with optional parameter overrides and custom time."""
    with model:
        setup = model_setup_ivp
        if time is not None:
            setup = xso.setup(solver='solve_ivp', model=model,
                              time=time, input_vars=BASE_INPUT_VARS)
        if i:
            setup = setup.xsimlab.update_vars(input_vars=i)
        model_out = setup.xsimlab.run()
    model_out['time'] = model_out.time.round(9)
    return model_out


def run_model_test_stability(i=None):
    """Steady-state + stability analysis with optional parameter overrides."""
    with model:
        if i:
            model_out = model_setup_stability.xsimlab.update_vars(input_vars=i).xsimlab.run()
        else:
            model_out = model_setup_stability.xsimlab.run()
    model_out['time'] = model_out.time.round(9)
    return model_out