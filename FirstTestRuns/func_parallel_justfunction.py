import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# import necessary packages
import numpy as np
import matplotlib.pyplot as plt
import xso

from func_parallel_justmodel import model, model_setup

def generate_iterable_parscan(parameter, par_range):
    return [{parameter:val} for val in par_range]
    

def make_run_model_test(model, model_setup):
    def run_model_test(i):
        with model:
            model_out = model_setup.xsimlab.update_vars(input_vars=i).xsimlab.run()
        model_out['time'] = model_out.time.round(9)
        return model_out
    return run_model_test

def run_model_test_v2(i):
    with model:
        model_out = model_setup.xsimlab.update_vars(input_vars=i).xsimlab.run()
    model_out['time'] = model_out.time.round(9)
    return model_out

        
def unpack_par_scan(iterable, data):
    var = list(iterable[0].keys())[0]
    i_tot=len(iterable)
    
    dat_out = []
    for dat,i, val in zip(data,range(i_tot),iterable) :
        dat_out.append(dat.assign_coords({var:list(val.values())[0]}).expand_dims(var))

    data_combined = xr.combine_by_coords(dat_out)
    
    return data_combined