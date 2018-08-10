'''
parameter grid for experimental setup
'''
import numpy as np

param_grid = {
        'random_state' : 10
        'num trials' : 5,
        'rows' : [10000],#, 25000, 50000, 100000],#, 100000,250000],
        'columns' : [10,50,100, 500],#, 1000],#, 100, 500, 1000],
        'sketch_factors' : 5,
        'density' : np.linspace(0.1,1.0, num=10)
    }
