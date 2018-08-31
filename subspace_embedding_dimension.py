'''
Experiment script to test subspace embedding dimension for various embedding
methods.
'''
import json
import itertools
import pickle
import helper
import numpy as np
import scipy as sp
from scipy import sparse
from scipy.sparse import load_npz
from timeit import default_timer
from lib import countsketch, srht, gaussian, classical_sketch
from lib import ClassicalSketch
import datasets_config
from joblib import Parallel, delayed
from my_plot_styles import plotting_params
from experiment_parameter_grid import param_grid
from synthetic_data_functions import generate_random_matrices


from matplotlib_config import update_rcParams
import matplotlib.pyplot as plt
from my_plot_styles import plotting_params


random_seed = 400
np.random.seed(random_seed)
sketch_names = ["CountSketch", "SRHT", "Gaussian"]
sketch_functions = {"CountSketch": countsketch.CountSketch,
                    "SRHT" : srht.SRHT,
                    "Gaussian" : gaussian.GaussianSketch}

n_trials = 5

def experiment_error_vs_sampling_factor(n, d, noise='gaussian',density=0.1):
    '''Measure the error as the sampling factor gamma is varied for the
    sketching dimension m = gamma*d where d is the dimensionality of the data.
    '''

    for dist in ['gaussian', 'cauchy']:
        print("Testing {} distribution".format(dis))
        A = generate_random_matrices(n,d,distribution=dist)
        true_covariance = A.T@A
        true_norm = np.linalg.norm(true_covariance,ord='fro')
        true_rank = np.linalg.matrix_rank(A)
        print("Rank of test matrix: {}".format(true_rank))
        sampling_factors = 1 + np.linspace(0.01,25.0,10)
        print(sampling_factors)
        sketch_dims = [np.int(sampling_factors[i]*d) for i in range(len(sampling_factors))]
        print(sketch_dims)
        # output dicts
        distortions = {sketch : {} for sketch in sketch_functions.keys()}
        #
        #
        print("Entering loop")
        for factor in sampling_factors:
            for sketch in sketch_functions.keys():
                #if sketch is "Gaussian":
                #    continue
                sketch_size = np.int(factor*d)
                error = 0
                rank_tests = np.zeros((n_trials,))
                for trial in range(n_trials):
                    print("{} distribution, testing sketch {} with sample factor (index) {}, trial: {}".format(dist, sketch, sketch_dims.index(sketch_size), trial))
                    summary = sketch_functions[sketch](data=A, sketch_dimension=sketch_size)
                    S_A = summary.sketch(A)
                    sketch_rank = np.linalg.matrix_rank(S_A)
                    print("Sketch rank {}".format(sketch_rank))
                    if sketch_rank == true_rank:
                        rank_tests[trial] = 1
                    approx_covariance = S_A.T@S_A
                    #approx_norm = np.linalg.norm(approx_covariance - true_covariance,ord='fro')
                    error += np.linalg.norm(true_covariance - S_A.T@S_A, ord='fro') / true_norm
                    #print("Approx ratio: {}".format(true_norm/approx_norm))
                    #print("Update val:{}".format(np.abs(approx_norm-true_norm) / true_norm))
                    #approx_factor += np.abs(approx_norm-true_norm)/true_norm
                distortions[sketch][factor] = error/n_trials
                num_fails = n_trials - np.sum(rank_tests)
                print("{} of the trials were rank deficient".format(num_fails))
        print(distortions)

        fig, ax = plt.subplots()

        for sketch in sketch_functions.keys():
            my_colour = plotting_params[sketch]['colour']
            my_marker = plotting_params[sketch]['marker']
            ax.plot(sampling_factors, distortions[sketch].values(), label=sketch, color=my_colour, marker=my_marker)
        ax.set_xlabel('Sampling factor')
        ax.set_ylabel('Distortion')
        ax.legend(title="{}".format(dist),frameon=False)
        plt.show()

    return distortions

def main():
    experiment_error_vs_sampling_factor(50000,20)

if __name__=='__main__':
    main()
