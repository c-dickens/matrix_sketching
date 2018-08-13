'''
parameter grid for experimental setup
'''
import numpy as np
from lib import countsketch, srht, gaussian

param_grid = {
        'random_state' : 10,
        'num trials' : 5,
        'rows' : [100000, 125000, 150000, 175000, 200000, 225000,250000, 300000, 400000,500000],#, 25000, 50000],#, 25000, 50000, 100000],#, 100000,250000],
        'columns' : [10,50,100,200,250,300, 400, 500, 600],#,350,400,450, 500],#, 1000],#, 100, 500, 1000],
        'sketch_factors' : 2,
        'density' : np.linspace(0.1,1.0, num=10)
    }

ihs_sketches = ["CountSketch", "SRHT"]
sketch_functions = {"CountSketch": countsketch.CountSketch,
                    "SRHT" : srht.SRHT,
                    "Gaussian" : gaussian.GaussianSketch}
