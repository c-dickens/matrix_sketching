'''
Real dataset subspace embedding experiments
'''
import json
import itertools
from pprint import PrettyPrinter
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

def subspace_embedding_check(test_data):
    '''Experiment to check at what point a subspace embedding is obtained
    on real datasets.'''
    all_datasets = datasets_config.datasets
    datasets = test_data
    results = {}
    n_trials = subspace_embedding_exp_setup['num trials']
    #

    for data in datasets:
        results[data] = {}


    # NB can just reuse the synthetic experiment here but replace the synthetic
    # distributions with real data.

    for data in datasets:
        print("-"*80)
        print("Testing dataset: {}".format(data))
        # input_file = datasets[data]["filepath"]
        input_file = all_datasets[data]['filepath']
        sparse_flag = False # a check to say if sparse data is found.
        # Defaults to False unles the sparse matrix can be read in.

        if data  is "kdd":
            print("Ignoring this one.")
            continue

        A = np.load(input_file)
        n,d = A.shape
        if (n,d) == (1,2):
            # this checks if the saved file is csr matrix and if so
            # continues the loop as the code needs to be different.
            print("SAVED IN SPARSE FORMAT SO SKIPPING FOR NOW.")
            continue
        Q,R = np.linalg.qr(A)
        lev_scores = np.linalg.norm(Q,axis=1,ord=2)*2
        coherence = np.max(lev_scores)
        least_lev = np.min(lev_scores)
        results[data]['coherence'] = coherence
        results[data]['min leverage score'] = least_lev
        print("Largest leverage score: {}".format(coherence))
        print("Least leverage score: {}".format(least_lev))
        print("Lev score deviation: {}".format(coherence/least_lev))
        true_covariance = A.T@A
        true_norm = np.linalg.norm(true_covariance,ord='fro')
        true_rank = np.linalg.matrix_rank(A)
        print("Rank of test matrix: {}".format(true_rank))
        sampling_factors = [1, 2, 5]
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
                    print("{} distribution, testing sketch {} with sample factor (index) {}, trial: {}".format(data, sketch, sketch_dims.index(sketch_size), trial))
                    rand_row_perm = np.random.permutation(A.shape[0])
                    summary = sketch_functions[sketch](data=A[rand_row_perm,:], sketch_dimension=sketch_size)
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
        results[data]['experiment'] =  distortions
    return results

def main():
        with open('figures/real_data_summary_time.json') as f:
            already_seen_data = json.load(f)
        print(already_seen_data.keys())
        saved_datasets = datasets_config.datasets.keys()
        print(saved_datasets)
        new_data_2_sketch= {}
        for dataset in saved_datasets:
            if dataset not in already_seen_data.keys():
                # i.e we have added a new datasets to test
                print(dataset)
                new_data_2_sketch[dataset] = {}

        print(new_data_2_sketch)
        new_exp_results = subspace_embedding_check(new_data_2_sketch)

        pretty = PrettyPrinter(indent=4)
        already_seen_data.update(new_exp_results)
        pretty.pprint(already_seen_data)
        with open('figures/real_data_summary_time.json', 'w') as outfile:
           json.dump(already_seen_data, outfile)


        # exp_results = subspace_embedding_check()
        # file_name = 'subspace_embedding_dimension_real_data'
        # np.save('figures/' + file_name + '.npy', exp_results)
        # with open('figures/' + file_name + '.json', 'w') as outfile:
        #     json.dump(exp_results, outfile)
        # print(exp_results)

if __name__ == "__main__":
    main()
