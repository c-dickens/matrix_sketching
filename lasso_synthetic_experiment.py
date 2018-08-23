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
from scipy.sparse import coo_matrix
from timeit import default_timer
from lib import countsketch, srht, classical_sketch
from lib import ClassicalSketch, IHS
import datasets_config
from joblib import Parallel, delayed
from synthetic_data_functions import generate_lasso_data
from my_plot_styles import plotting_params
from experiment_parameter_grid import param_grid, ihs_sketches, sketch_functions


######################  HELPER FUNCTIONS    ####################################
def original_lasso_objective(X,y, regulariser,x, penalty=False):

    if penalty:
        return 0.5*np.linalg.norm(X@x-y,ord=2)**2 + regulariser*np.linalg.norm(x,1)
    else:
        return 0.5*np.linalg.norm(X@x-y,ord=2)**2

def sklearn_wrapper(X,y,n,d, regulariser, trials):
    clf = Lasso(regulariser)
    lasso_time = 0
    for i in range(trials):
        print("Trial ", i)
        lasso_start = default_timer()
        lasso = clf.fit(n*X,n*y)
        lasso_time += default_timer() - lasso_start
        x_opt = lasso.coef_
        f_opt = original_lasso_objective(X,y,regulariser,x_opt,penalty=True)
    return x_opt, f_opt, lasso_time/trials



################################################################################

def experiment_sklearn_vs_sketch_time_d(n):
    '''Fix an n and generate datasets of varying width in order to see how the
    LASSO problem scales with respect to d'''

    np.random.seed(param_grid["random_state"])
    cols = param_grid['columns']
    sketch_factor = param_grid['sketch_factors']
    trials = param_grid['num trials']
    sklearn_lasso_bound = 10

    lasso_time = 0

    # results dicts
    results = {}
    results["sklearn"] = {}
    for d in cols:
        results["sklearn"][d] = {}
    for sketch in ihs_sketches:
        results[sketch] = {}
        for d in cols:
            results[sketch][d] = {}
    print(results)

    for d in cols:
        print("*"*80)
        print("Generating data with {} rows and {} columns".format(n,d))
        X, dense_X, y,truth = generate_lasso_data(n,d, data_density=0.1,sigma=1.0,)
        #print('REQUIRED ITERATIONS {}'.format(1+np.int(np.ceil(np.log(n)))))
        print("Converting to COO format")
        #sparse_data = coo_matrix(X)
        rows, cols, vals = X.row, X.col, X.data
        print("Beginning experiment")

        for method in results.keys():
            print(method)



            if method is "sklearn":
                x_opt, f_opt, lasso_time = sklearn_wrapper(X,y,n,d, sklearn_lasso_bound, 1)
                results["sklearn"][d] = {#"estimator"        : x_opt,
                                         "error to truth"   : np.linalg.norm(X@(x_opt-truth),ord=2)**2,
                                         "solve time"       : lasso_time,
                                         "objective value"  : f_opt}


            else:
                total_setup_time, total_sketch_time, total_opt_time, total_n_iters = 0,0,0,0
                sklearn_error, truth_error, total_obj_val = 0,0,0
                x_hat = np.zeros((d,))
                for i in range(trials):
                    print("Trial ", i)
                    total_iters = 0
                    #print("Testing {} iterations for IHS".format(1+np.int(np.ceil(np.log(n)))))
                    ihs_lasso = IHS(data=dense_X, targets=y, sketch_dimension=sketch_factor*d,
                                    sketch_type=method,number_iterations=np.int(np.ceil(np.log10(n))),
                                    data_rows=rows,data_cols=cols,data_vals=vals,
                                    random_state=param_grid["random_state"])
                    x0, setup_time, sketch_time, opt_time, n_iters = ihs_lasso.fast_solve({'problem' : "lasso", 'bound' : sklearn_lasso_bound}, timing=True)
                    print("Sketch time on ({},{}) is {}".format(n,d,sketch_time))
                    print("N iters required {}".format(n_iters))
                    sklearn_error += np.linalg.norm(X@(x0-x_opt),ord=2)**2
                    truth_error += np.linalg.norm(X@(x0-truth),ord=2)**2
                    total_obj_val += original_lasso_objective(X,y,sklearn_lasso_bound,x0)
                    total_n_iters += n_iters
                    total_setup_time += setup_time
                    total_sketch_time += sketch_time
                    total_opt_time += opt_time

                mean_n_iters = np.int(np.ceil(total_n_iters/trials))

                mean_setup_time = total_setup_time / trials
                mean_sketch_time = total_sketch_time / trials
                mean_opt_time = total_opt_time / trials
                mean_sklearn_error = sklearn_error / trials
                mean_truth_error = truth_error / trials
                mean_obj_val = total_obj_val / trials



                results[method][d] = {#"estimate"           : x0,
                                      "error to sklearn"   : mean_sklearn_error,
                                      "error to truth"     : mean_truth_error,
                                      "objective val"      : mean_obj_val, #original_lasso_objective(X,y,sklearn_lasso_bound,x0),
                                      "setup time"         : mean_setup_time, #mean_setup_time,
                                      "sketch_time"        : mean_sketch_time, #mean_sketch_time,
                                      "optimisation time"  : mean_opt_time, #mean_opt_time,
                                      "total time"         : mean_setup_time + mean_sketch_time + mean_opt_time, #mean_setup_time+mean_sketch_time+mean_opt_time,
                                      "num iters"          : mean_n_iters,
                                      "num columns"        : d}


    file_name = 'figures/lasso_synthetic_times_vary_d_at_n_' + str(n) + ".npy"
    np.save(file_name, results)
    print(json.dumps(results,indent=4))

def experiment_sklearn_vs_sketch_n(d):
    '''
    Fix a d and test various n values for the large n lasso regression case
    '''
    np.random.seed(param_grid["random_state"])
    rows = param_grid['rows']
    sketch_factor = param_grid['sketch_factors']
    trials = param_grid['num trials']
    sklearn_lasso_bound = 10

    lasso_time = 0

    # results dicts
    results = {}
    results["sklearn"] = {}
    # for n in rows:
    #     results["sklearn"][n] = {}
    for sketch in ihs_sketches:
        results[sketch] = {}
        for n in rows:
            results[sketch][n] = {}
    print(results)

    for n in rows:
        print("*"*80)
        print("Generating data with {} rows".format(n))
        X,y,truth = generate_lasso_data(n,d,sigma=1.0,density=0.2)
        print("Converting to COO format")
        sparse_data = coo_matrix(X)
        rows, cols, vals = sparse_data.row, sparse_data.col, sparse_data.data
        print("Beginning experiment")

        for method in results.keys():
            print(method)

            if method is "sklearn":
                x_opt, f_opt, lasso_time = sklearn_wrapper(X,y,n,d, sklearn_lasso_bound, trials)
                results["sklearn"][n] = {#"estimator"        : x_opt,
                                         "error to truth"   : np.linalg.norm(X@(x_opt-truth),ord=2)**2,
                                         "solve time"       : lasso_time,
                                         "objective value"  : f_opt}
            else:
                ihs_lasso = IHS(data=X, targets=y, sketch_dimension=sketch_factor*d,
                                sketch_type=method,number_iterations=1+np.int(np.ceil(np.log(n))),
                                data_rows=rows,data_cols=cols,data_vals=vals,
                                random_state=param_grid["random_state"])
                x0, setup_time, sketch_time, opt_time, n_iters = ihs_lasso.fast_solve({'problem' : "lasso", 'bound' : sklearn_lasso_bound}, timing=True)
                results[method][n] = {#"estimate"           : x0,
                                      "error to sklearn"   : np.linalg.norm(X@(x0-x_opt),ord=2)**2,
                                      "error to truth"     : np.linalg.norm(X@(x0-truth),ord=2)**2,
                                      "objective val"      : original_lasso_objective(X,y,sklearn_lasso_bound,x0),
                                      "setup time"         : setup_time,
                                      "sketch_time"        : sketch_time,
                                      "optimisation time"  : opt_time,
                                      "total time"         : setup_time+sketch_time+opt_time,
                                      "num iters"          : n_iters,
                                      "num columns"        : d}


    file_name = 'figures/lasso_synthetic_times_vary_n_at_d_' + str(d) + ".npy"
    np.save(file_name, results)
    print(json.dumps(results,indent=4))


def main():
    #experiment_sklearn_vs_sketch_time_d()

    for n in param_grid['rows']:
        experiment_sklearn_vs_sketch_time_d(n)

    #experiment_sklearn_vs_sketch_n(200)

if __name__ == "__main__":
    main()
