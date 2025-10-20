import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# import necessary packages
import numpy as np
import matplotlib.pyplot as plt
import xso

from Stocketal2008_comps import (Nutrient, PhytoSizeSpectrum, ZooSizeSpectrum, 
    ConstantExternalNutrient, LinearForcingInput, 
    MonodGrowth_SizeBased, 
    StockGrazingMatrix, Stock_GGE_MatrixGrazing, 
    StockPhytoMortality, StockZooMortality, StockZooMortality_simpleinput)



# number size classes of phytoplankton and zooplankton
P_num = 3
Z_num = 4

# create initial biomass
phyto_init = np.tile(1.5, (P_num))
zoo_init = np.tile(.1, (Z_num))

# calculate log-spaced size classes from ranges and total number
phyto_sizes = [0.63, 6.3, 63]
zoo_sizes = [6.3, 63, 630, 6300]

# Phytoplankton parameters
phyto_ks = [0.062, 0.45, 3.3]
phyto_mu0 = [1.0, 1.26, 0.42]
phyto_mortality = [1.0, 0, 0]
phyto_mort_exponent = [2.0, 1.0, 1.0]
phyto_recycling = [1.0, 0.0, 0.0]

# Zooplankton parameters
zoo_imax = [10.0, 3.3, 1.1, 0.36]
zoo_Ki = 3.0
zoo_frac_assim = 0.7  # alpha
zoo_frac_excreted = 0.45  # R
zoo_gge = 0.25
# prey availability: basically just 1 for size class below of Z and P, no other grazing! issa matrix 4 x 7
zoo_prey_avail = np.array([[1, 0, 0, 0, 0, 0, 0], # Z1
                          [0, 1, 0, 1, 0, 0, 0], # Z2
                          [0, 0, 1, 0, 1, 0, 0], # Z3
                          [0, 0, 0, 0, 0, 1, 0]]) # Z4

# dens dep prey exploitation factor
zoo_frac_egest_recycled = [1, 1, 0, 0]

# zoo mortality
zoo_higherordermortality = 0.0093
zoo_mortality_array = [0, 0, 0, zoo_higherordermortality]
zoo_mort_exponent = [0, 0, 0, 1]
zoo_frac_mortylity_recycled = [0, 0, 0, 0.5]



nutrient_input = 0.0053 # 0.017



model = xso.create({
            # State variables
            'Nutrient': Nutrient,
            'Phytoplankton': PhytoSizeSpectrum,
            'Zooplankton': ZooSizeSpectrum,
        
            # Flows:
            'Inflow': LinearForcingInput,
        
            # Growth
            'Growth': MonodGrowth_SizeBased,
        
            # Grazing
            'Grazing': StockGrazingMatrix,
            'GGE': Stock_GGE_MatrixGrazing,
        
            # Mortality
            'PhytoMortality': StockPhytoMortality,
            'HigherOrderMortality': StockZooMortality_simpleinput,
        
            # Forcings
            'N0': ConstantExternalNutrient,
        })
        
model_setup = xso.setup(solver='solve_ivp', model=model,
        time=np.arange(0,5000),
        input_vars={
                # State variables
                'Nutrient':{'value_label':'N','value_init':1.0},
                'Phytoplankton':{'biomass_label':'P','biomass_init':phyto_init, 'phyto_index':phyto_sizes},
             
                'Zooplankton':{'biomass_label':'Z','biomass_init':zoo_init, 'zoo_index': zoo_sizes},
            
                # Flows:
                'Inflow':{'forcing':'N0', 'rate':1., 'var':'N'},
            
                # Growth
                'Growth':{'resource':'N', 'consumer':'P', 'halfsat':phyto_ks, 'mu_max':phyto_mu0},

                # Grazing
                'Grazing':{'resource':'P', 'consumer':'Z', 'Imax':zoo_imax, 'KsZ':zoo_Ki, 'phiPZ':zoo_prey_avail},
                'GGE':{'grazed_phyto':'P', 'grazed_zoo':'Z', 'assimilated_consumer':'Z', 'egested_detritus':'N', 
                       'R':zoo_frac_excreted, 'alpha':zoo_frac_assim, 'f_I':zoo_frac_egest_recycled, 'gge':zoo_gge},
            
                # Mortality
                'PhytoMortality':{'population':'P', 'nutrient':'N', 'rate':phyto_mortality, 'exponent':phyto_mort_exponent, 'recycling':phyto_recycling},
                'HigherOrderMortality':{'population':'Z', 'nutrient':'N', 'rate':zoo_higherordermortality, 'exponent':zoo_mort_exponent, 'recycling':zoo_frac_mortylity_recycled},
 # Forcings
                'N0':{'forcing_label':'N0', 'value':nutrient_input},
        })



def run_model_test(i):
    with model:
        model_out = model_setup.xsimlab.update_vars(input_vars=i).xsimlab.run()
        
    # solve ivp introduces rounding errors for time, which mess up combining datasets later: hence rounding
    model_out['time']= model_out.time.round(9)
    return model_out