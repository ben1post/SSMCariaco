import numpy as np
import matplotlib.pyplot as plt
import xso

from phydra.models import NPxZxSizeBased

def calculate_sizes(size_min, size_max, num):
    """initializes log spaced array of sizes from ESD size range"""
    numbers = np.array([i for i in range(num)])
    sizes = (np.log(size_max) - np.log(size_min))* numbers / (num-1) + np.log(size_min)
    return np.exp(sizes)

def calculate_zoo_I0(sizes):
    """initializes allometric Zooplankton ingestion rate based on array of sizes (ESD)"""
    return 26 * sizes ** -0.4

def calculate_phyto_mu0(sizes):
    """initializes allometric Phytoplankton maximum growth rate based on array of sizes (ESD)
    allometric relationships are taken from meta-analyses of lab data"""
    return 2.6 * sizes ** -0.45

def calculate_phyto_ks(sizes):
    """initializes allometric Phytoplankton half-saturation constant based on array of sizes (ESD)"""
    return sizes * .1

def calculate_opt_size(sizes):
    """Calculating optimal prey size from Zooplankton sizes"""
    return 0.65 * sizes ** 0.56


def init_phiP(phytosize, zoopreyoptsize):
    """creates matrix of feeding preferences [P...P10] for each [Z]"""
    phiP = np.array([[np.exp(-((np.log10(xpreyi) - np.log10(xpreyoptj)) / 0.25) ** 2)
                      for xpreyi in phytosize] for xpreyoptj in zoopreyoptsize])
    return phiP



# number size classes of phytoplankton and zooplankotn
PZ_num = 10

# create initial biomass
phyto_init = np.tile(.5/PZ_num, (PZ_num))
zoo_init = np.tile(.1/PZ_num, (PZ_num))

# calculate log-spaced size classes from ranges and total number
phyto_sizes = calculate_sizes(1.,50.,PZ_num)
zoo_sizes = 2.16 * phyto_sizes **1.79 

# ingestion
zoo_I0 = calculate_zoo_I0(zoo_sizes)

# growth
phyto_mu0 = calculate_phyto_mu0(phyto_sizes)
phyto_ks = calculate_phyto_ks(phyto_sizes)

# grazing
preyoptsize = calculate_opt_size(zoo_sizes)
phiP = init_phiP(phyto_sizes, preyoptsize)


BASE_INPUT_VARS={
        # State variables
        'Nutrient':{'value_label':'N','value_init':1.},
        'Phytoplankton':{'biomass_label':'P','biomass_init':phyto_init, 'phyto_index':phyto_sizes},
        'Zooplankton':{'biomass_label':'Z','biomass_init':zoo_init, 'zoo_index': zoo_sizes},
    
        # Flows:
        'Inflow':{'forcing':'N0', 'rate':1., 'var':'N'},
    
        # Growth
        'Growth':{'resource':'N', 'consumer':'P', 'halfsat':phyto_ks, 'mu_max':phyto_mu0},

        # Grazing
        'Grazing':{'resource':'P', 'consumer':'Z',
                   'Imax':zoo_I0, 'KsZ':3., 'phiP':phiP},
        'GGE':{'grazed_resource':'P', 'assimilated_consumer':'Z', 'egested_detritus':'N', 
               'epsilon':1./3., 'f_eg':1./3.},
    
        # Mortality
        'PhytoMortality':{'var':'P', 'rate':0.1*phyto_mu0},
        'ZooMortality':{'var':'Z', 'rate':1.},

        # Forcings
        'N0':{'forcing_label':'N0', 'value':1.},
}


model = NPxZxSizeBased
        
model_setup_stability = xso.setup(
    solver='stability', 
    model=model,
    time=[0,1],
    input_vars=BASE_INPUT_VARS
)

model_setup_ivp = xso.setup(
    solver='solve_ivp', 
    model=model,
    time=np.arange(0,365*10),
    input_vars=BASE_INPUT_VARS
)
