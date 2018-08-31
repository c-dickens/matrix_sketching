'''
parameter grid for experimental setup
'''
import numpy as np
from lib import countsketch, srht, gaussian

param_grid = {
        'random_state' : 10,
        'num trials' : 5,
        'rows' : [50_000],# 500000, 1000000],
        'columns' : [10,50,100,200],#, 250, 300, 350, 400],#,
        'sketch_factors' : 2,
        'density' : np.linspace(0.05,1.0, num=10)
    }

ihs_sketches = ["CountSketch", "SRHT"]
sketch_functions = {"CountSketch": countsketch.CountSketch,
                    "SRHT" : srht.SRHT,
                    "Gaussian" : gaussian.GaussianSketch}
