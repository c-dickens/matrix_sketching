'''
Real dataset subspace embedding experiments
'''
import json
import itertools
import numpy as np
import scipy as sp
from scipy import sparse
from scipy.sparse import load_npz, coo_matrix
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from timeit import default_timer
from lib import countsketch, srht, gaussian, classical_sketch
from lib import ClassicalSketch, IHS
from lasso_synthetic_experiment import sklearn_wrapper, original_lasso_objective
import datasets_config
from joblib import Parallel, delayed
from my_plot_styles import plotting_params
from experiment_parameter_grid import param_grid, subspace_embedding_exp_setup

def subspace_embedding_check():
    '''Experiment to check at what point a subspace embedding is obtained.'''
    datasets = datasets_config.datasets
    sketch_factors = [200]

    # NB can just reuse the synthetic experiment here but replace the synthetic
    # distributions with real data.
