'''
Experiment to compare wall-clock time vs accuracy of the ihs method with
different sketches and different sketch sizes
'''
import itertools
from pprint import PrettyPrinter
import numpy as np
import scipy as sp
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.datasets import make_regression
from time import process_time
from utils import *
from lib import iterative_hessian_sketch as ihs
from synthetic_data_functions import my_lasso_data
from experiment_parameter_grid import time_error_ihs_grid, ihs_sketches

import datetime
from datetime import timedelta

from matplotlib_config import update_rcParams
import matplotlib.pyplot as plt
from my_plot_styles import plotting_params




def error_vs_time(n,d,sampling_factors,trials,times):
    '''Show that a random lasso instance is approximated by the
    hessian sketching scheme'''
    print(80*"-")
    print("TESTING LASSO ITERATIVE HESSIAN SKETCH ALGORITHM")
    file_name = 'figures/ihs_fixed_time_' + str(n) + '_' + str(d) + '.npy'
    sklearn_lasso_bound = 5.0
    times2test = times


    print("Generating  data")
    X,y,x_star = my_lasso_data(n,d)
    X = normalize(X)
    print("Converting to CSR format")
    sparse_data = sparse.csr_matrix(X)
    print("*"*80)


    ### Test Sklearn implementation
    print("Beginning test")
    clf = Lasso(sklearn_lasso_bound)
    x_opt, f_opt, sklearn_time = sklearn_wrapper(X,y,n,d,sklearn_lasso_bound, trials)
    print("LASSO-skl time: {}".format(sklearn_time))

    # ground Truths
    sklearn_error2truth = prediction_error(X,x_opt, x_star)



    time_results = {"Sklearn" : {"error to truth" : sklearn_error2truth,
                                 "objective"      : f_opt,
                                 "solve time"     : sklearn_time},}

    for sketch in ihs_sketches:
        time_results[sketch] = {}
        for gamma in sampling_factors:
            time_results[sketch][gamma] = {}

    for sketch_method in ihs_sketches:
        for gamma in sampling_factors:
            sketch_size = np.int(gamma*d)
            for time in times2test:

                total_error2opt       = 0
                total_error2truth     = 0
                total_sol_error       = 0
                total_objective_error = 0
                total_iters_used      = 0

                print("IHS-LASSO ALGORITHM on ({},{}) WITH {}, gamma {}".format(n,d,sketch_method, gamma), 60*"*")
                print("Testing time: {}".format(time))
                funct_dict = {'time_to_run' : time, 'problem' : "lasso", 'bound' : sklearn_lasso_bound}
                for _trial in range(trials):
                    shuffled_ids = np.random.permutation(n)
                    X_train, y_train = X[shuffled_ids,:], y[shuffled_ids]
                    sparse_X_train = sparse_data[shuffled_ids,:]
                    sparse_X_train = sparse_X_train.tocoo()
                    rows, cols, vals = sparse_X_train.row, sparse_X_train.col, sparse_X_train.data
                    ihs_lasso = ihs.IHS(data=X_train, targets=y_train, sketch_dimension=sketch_size,
                                        sketch_type=sketch_method,
                                        number_iterations=10,
                                        data_rows=rows,data_cols=cols,data_vals=vals)


                    #x_ihs,_, _, _, iters_used = ihs_lasso.fast_solve({'problem' : "lasso", 'bound' : sklearn_lasso_bound},timing=True)
                    x_ihs,iters_used = ihs_lasso.solve_for_time(**funct_dict)

                    # Update dict output values
                    error2opt = prediction_error(X,x_opt,x_ihs)**2
                    solution_error = (1/n)*np.linalg.norm(x_ihs - x_opt)**2

                    # Update counts
                    total_error2opt += error2opt
                    total_sol_error += solution_error
                    total_iters_used += iters_used

                total_error2opt /= trials
                total_sol_error /= trials
                total_iters_used /= trials

                print("Mean ||x^* - x'||^2: {}".format(total_sol_error))
                print("Mean number of {} iterations used".format(total_iters_used))
                time_results[sketch_method][gamma][time] = {"error to opt" : total_error2opt,
                                                     "solution error" : total_sol_error,
                                                     "num iterations" : total_iters_used}

    pretty = PrettyPrinter(indent=4)
    pretty.pprint(time_results)
    np.save(file_name, time_results)

    # fig, ax = plt.subplots()
    # for sketch_method in ihs_sketches:
    #     for gamma in sampling_factors:
    #         yvals = [np.log10(time_results[sketch_method][gamma][time]["error to opt"]) for time in times2test]
    #         ax.plot(times2test, yvals, label=sketch_method+str(gamma))
    # ax.axvline(sklearn_time,color='k',label='Sklearn')
    # ax.legend()
    # ax.set_ylabel("log(error))")
    # ax.set_xlabel('log(Time(seconds))')
    # ax.set_xscale('log')
    # plt.show()
    pass



def main():
    np.random.seed(time_error_ihs_grid['random_state'])
    sampling_factors = time_error_ihs_grid['sketch_factors']
    n_trials = time_error_ihs_grid['num trials']
    time_range = time_error_ihs_grid['times']
    # for n,d in itertools.product(time_error_ihs_grid['rows'],time_error_ihs_grid['columns']):
    for n,d in itertools.product([50_000],[50]):
        print("Testing n={}, d={}".format(n,d))
        error_vs_time(n,d,sampling_factors,n_trials, time_range)

        # Parallel(n_jobs=-1)(delayed(error_vs_time)\
        #                 (n,d,sampling_factors,n_trials) for d in time_error_ihs_grid['columns'])

    pass

if __name__ == "__main__":
    main()
