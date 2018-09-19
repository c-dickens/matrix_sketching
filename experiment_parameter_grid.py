'''
parameter grid for experimental setup
'''
import numpy as np
from lib import countsketch, srht, gaussian

param_grid = {
        'random_state' : 10,
        'num trials' : 5,
        'rows' : [25_000],# 50_000, 75_000],# 100_000, 125_000],# 50_000],#, 100_000, 50_000 ],# 500000, 1000000],
        'columns' : [10, 50, 100, 1000, 5000],# 100]#,1000,5000],#, 250, 300, 350, 400],#,
        'sketch_factors' : [1.1, 2,5],
        'density' : [0.01, 0.05, 0.1 , 0.2 , 0.3 , 0.4, 0.6, 0.8, 1.0] #np.linspace(0.01,1.0, num=10)
    }

subspace_embedding_exp_setup = {
    'random_state' : 400,
    'num trials'   : 10,
    'aspect ratio range' : np.linspace(0.1,0.5, 5),
    'rows'               : [2**11]
}

ihs_sketches = ["CountSketch", "SRHT"]
sketch_functions = {"CountSketch": countsketch.CountSketch,
                    "SRHT" : srht.SRHT,
                    "Gaussian" : gaussian.GaussianSketch}
sketch_names = ["CountSketch", "SRHT", "Gaussian"]
