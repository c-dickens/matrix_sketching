'''
parameter grid for experimental setup
'''
import numpy as np
from lib import countsketch, srht, gaussian

param_grid = {
        'random_state' : 10,
        'num trials' : 3,
        'rows' : [100000, 200000, 300000, 400000, 500000],# 500000, 1000000],
        'columns' : [10,50,100,200,300, 400, 500],#,350,400,450, 500],#, 1000],#, 100, 500, 1000],
        'sketch_factors' : 2,
        'density' : np.linspace(0.05,1.0, num=10)
    }

ihs_sketches = ["CountSketch", "SRHT"]
sketch_functions = {"CountSketch": countsketch.CountSketch,
                    "SRHT" : srht.SRHT,
                    "Gaussian" : gaussian.GaussianSketch}
