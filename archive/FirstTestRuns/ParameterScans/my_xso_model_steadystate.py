import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import xso

from phydra.models import NPChemostat as model

model_setup_ivp = xso.setup(solver='solve_ivp', model=model,
            time=np.arange(0,2000, 0.1),
            input_vars={
                    # State variables
                    'Nutrient':{'value_label':'N','value_init':1.},
                    'Phytoplankton':{'value_label':'P','value_init':0.1},
                
                    # Flows:
                    'Inflow':{'source':'N0', 'rate':0.1, 'sink':'N'},
                    'Outflow':{'var_list':['N', 'P'], 'rate':0.1},
                
                    # Growth
                    'Growth':{'resource':'N', 'consumer':'P', 'halfsat':0.7, 'mu_max':1},
                
                    # Forcings
                    'N0':{'forcing_label':'N0', 'value':1.}
            })

model_setup = xso.setup(solver='fsolve', model=model,
            time=[0,1],
            input_vars={
                    # State variables
                    'Nutrient':{'value_label':'N','value_init':1.},
                    'Phytoplankton':{'value_label':'P','value_init':0.1},
                
                    # Flows:
                    'Inflow':{'source':'N0', 'rate':0.1, 'sink':'N'},
                    'Outflow':{'var_list':['N', 'P'], 'rate':0.1},
                
                    # Growth
                    'Growth':{'resource':'N', 'consumer':'P', 'halfsat':0.7, 'mu_max':1},
                
                    # Forcings
                    'N0':{'forcing_label':'N0', 'value':1.}
            })

model_setup_deriv = xso.setup(solver='deriv', model=model,
            time=[0,1],
            input_vars={
                    # State variables
                    'Nutrient':{'value_label':'N','value_init':1.},
                    'Phytoplankton':{'value_label':'P','value_init':0.1},
                
                    # Flows:
                    'Inflow':{'source':'N0', 'rate':0.1, 'sink':'N'},
                    'Outflow':{'var_list':['N', 'P'], 'rate':0.1},
                
                    # Growth
                    'Growth':{'resource':'N', 'consumer':'P', 'halfsat':0.7, 'mu_max':1},
                
                    # Forcings
                    'N0':{'forcing_label':'N0', 'value':1.}
            })