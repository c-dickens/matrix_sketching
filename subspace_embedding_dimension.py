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
from experiment_parameter_grid import subspace_embedding_exp_setup, sketch_names, sketch_functions
from synthetic_data_functions import generate_random_matrices


from matplotlib_config import update_rcParams
import matplotlib.pyplot as plt
from my_plot_styles import plotting_params


random_seed = subspace_embedding_exp_setup['random_state']
np.random.seed(random_seed)
# sketch_names = ["CountSketch", "SRHT", "Gaussian"]
# sketch_functions = {"CountSketch": countsketch.CountSketch,
#                     "SRHT" : srht.SRHT,
#                     "Gaussian" : gaussian.GaussianSketch}



def experiment_error_vs_sampling_factor(n, d, noise='gaussian',density=0.1):
    '''Measure the error as the sampling factor gamma is varied for the
    sketching dimension m = gamma*d where d is the dimensionality of the data.
    '''
    dists_2_test = ['cauchy','power', 'gaussian', 'exponential', 'uniform']
    results = {}
    n_trials = subspace_embedding_exp_setup['num trials']

    for dist_name in dists_2_test:
        results[dist_name] = {}

    # Other miscellaneios information which might need to be saved
    results['num trials'] = n_trials
    results['aspect ratio'] = d/n

    for dist in dists_2_test:
        print("Testing {} distribution".format(dist))
        A = generate_random_matrices(n,d,distribution=dist)
        Q,R = np.linalg.qr(A)
        lev_scores = np.linalg.norm(Q,axis=1,ord=2)*2
        coherence = np.max(lev_scores)
        least_lev = np.min(lev_scores)
        results[dist]['coherence'] = coherence
        results[dist]['min leverage score'] = least_lev
        print("Largest leverage score: {}".format(coherence))
        print("Least leverage score: {}".format(least_lev))
        print("Lev score deviation: {}".format(coherence/least_lev))
        true_covariance = A.T@A
        true_norm = np.linalg.norm(true_covariance,ord='fro')
        true_rank = np.linalg.matrix_rank(A)
        print("Rank of test matrix: {}".format(true_rank))
        sampling_factors = 1 + np.linspace(0.025,0.125,5)
        print(sampling_factors)
        sketch_dims = [np.int(sampling_factors[i]*d) for i in range(len(sampling_factors))]
        print(sketch_dims)
        # output dicts
        distortions = {sketch : {} for sketch in sketch_functions.keys()}

        for factor in sampling_factors:
            for sketch in sketch_functions.keys():
                sketch_size = np.min([np.int(factor*d), n])
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
                    error += np.linalg.norm(true_covariance - S_A.T@S_A, ord='fro') / true_norm
                num_fails = n_trials - np.sum(rank_tests)
                distortions[sketch][factor] = {'mean distortion': error/n_trials,
                                                'rank failures' : num_fails}

                print("{} of the trials were rank deficient".format(np.int(num_fails)))
        results[dist]['experiment'] =  distortions
    return results

def main():

    range_to_test = subspace_embedding_exp_setup['aspect ratio range'] #np.concatenate((np.linspace(0.01,0.1,10),np.linspace(0.125, 0.5,4)))
    for n in subspace_embedding_exp_setup['rows']:
        for scale in range_to_test:
            d = np.int(scale*n)
            print("Testing {} and {}".format(n,d))
            exp_results = experiment_error_vs_sampling_factor(n,d)
            file_name = 'subspace_embedding_dimension_' + str(n) + "_" + str(d)
            np.save('figures/' + file_name + '.npy', exp_results)
            with open('figures/' + file_name + '.json', 'w') as outfile:
                json.dump(exp_results, outfile)
            print(exp_results)

if __name__=='__main__':
    main()
