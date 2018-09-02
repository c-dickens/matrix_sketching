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

    for dist in ['cauchy','power', 'gaussian', 'exponential', 'uniform']:
        print("Testing {} distribution".format(dist))
        A = generate_random_matrices(n,d,distribution=dist)
        Q,R = np.linalg.qr(A)
        lev_scores = np.linalg.norm(Q,axis=1,ord=2)*2
        coherence = np.max(lev_scores)
        least_lev = np.min(lev_scores)
        print("Largest leverage score: {}".format(coherence))
        print("Least leverage score: {}".format(least_lev))
        print("Lev score deviation: {}".format(coherence/least_lev))
        true_covariance = A.T@A
        true_norm = np.linalg.norm(true_covariance,ord='fro')
        true_rank = np.linalg.matrix_rank(A)
        print("Rank of test matrix: {}".format(true_rank))
        sampling_factors = 1 + np.linspace(0.05,0.2,3)
        print(sampling_factors)
        sketch_dims = [np.int(sampling_factors[i]*d) for i in range(len(sampling_factors))]
        print(sketch_dims)
        # output dicts
        distortions = {sketch : {} for sketch in sketch_functions.keys()}

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
                num_fails = n_trials - np.sum(rank_tests)
                distortions[sketch][factor] = {'mean distortion': error/n_trials,
                                                'rank failures' : num_fails}

                #distortions[sketch][factor]['num rank fails'] = num_fails
                print("{} of the trials were rank deficient".format(np.int(num_fails)))
        print(distortions)


        fig, ax = plt.subplots(figsize=(12,8))

        for sketch in sketch_functions.keys():
            my_colour = plotting_params[sketch]['colour']
            my_marker = plotting_params[sketch]['marker']
            my_line = plotting_params[sketch]['line_style']
            my_title = dist.capitalize()
            #print(distortions[sketch])
            #print("Vals ", distortions[sketch].values())
            x_vals = np.array(list(distortions[sketch].keys())) #sampling factor for x axis
            y_vals = np.array([distortions[sketch][key]['mean distortion'] for key in x_vals])
            # ax.scatter(x_vals, y_vals)

            # rank_fail_check[i] is 1 if sample factor i gave a rank fail
            rank_fail_check = np.array([distortions[sketch][key]['rank failures'] for key in x_vals])
            bad_ids = rank_fail_check[rank_fail_check > 0] #np.where(rank_fail_check > 0)
            bad_x = x_vals[rank_fail_check > 0]
            bad_y = y_vals[rank_fail_check > 0]
            good_x = x_vals[rank_fail_check == 0]
            good_y = y_vals[rank_fail_check == 0]
            my_marker_size = [30*i for i in bad_ids]
            ax.scatter(bad_x, bad_y, color=my_colour, marker='x', s=my_marker_size)
            ax.scatter(good_x, good_y, color=my_colour, marker=my_marker)
            ax.plot(x_vals, y_vals, color=my_colour, linestyle=my_line, label=sketch)
            ax.legend(title=my_title,frameon=False)
        ax.set_xlabel('Sampling factor ($\gamma$)')
        ax.set_ylabel('Distortion ($\epsilon$)')
        ax.set_ylim(bottom=0)
        ax.grid()
        # #
        plt.show()

    return distortions

def main():
    experiment_error_vs_sampling_factor(2500,1000)

if __name__=='__main__':
    main()
