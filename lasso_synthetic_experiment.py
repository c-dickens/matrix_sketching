'''Experiment to see how the CountSketch fares when in solving the
overconstrained LASSO problem.

1. Increase n with a fixed number of columns and measure time.'''
import json
import itertools
import pickle
import helper
import numpy as np
import scipy as sp
from sklearn.linear_model import Lasso
from scipy import sparse
from scipy.sparse import load_npz
from timeit import default_timer
from lib import countsketch, srht, gaussian, classical_sketch
from lib import ClassicalSketch
import datasets_config
from joblib import Parallel, delayed
from synthetic_data_functions import generate_lasso_data
from my_plot_styles import plotting_params
from experiment_parameter_grid import param_grid


def experiment_sklearn_vs_countsketch_time_n():
    np.random.seed(param_grid["random_state"])
    n = param_grid['rows']
    cols = param_grid['columns']
    sketch_factor = param_grid['sketch_factors']
    sklearn_lasso_bound = 10
    trials = 5
    lasso_time = 0





    for d in cols:
        print("Generating data with {} columns".format(d))
        X,y,truth = generate_lasso_data(n,d,sigma=1.0,density=0.2)
        print("Converting to COO format")
        sparse_data = coo_matrix(X)
        rows, cols, vals = sparse_data.row, sparse_data.col, sparse_data.data
        print("Beginning experiment")
        lasso_estimator = Lasso()

        for i in range(trials):
            lasso_start = default_timer()
            X, y, coef = generate_lasso_data(nrows, ncols, sigma=1.0 density=0.25)
            lasso_time += default_timer() - lasso_start
        print("LASSO-skl time: {}".format(lasso_time/trials))



def main():



if __nam__ == "__main__":
    main()
