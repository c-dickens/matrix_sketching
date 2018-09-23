'''
Experiment to compare wall-clock time vs accuracy of the ihs method with
different sketches and different sketch sizes
'''
import numpy as np
import scipy as sp
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.datasets import make_regression
from time import process_time
from utils import *
from lib import iterative_hessian_sketch as ihs
from synthetic_data_functions import generate_lasso_data
from experiment_parameter_grid import param_grid, sketch_names, sketch_functions, ihs_sketches

import datetime
from datetime import timedelta

from matplotlib_config import update_rcParams
import matplotlib.pyplot as plt
from my_plot_styles import plotting_params


def my_lasso_data(m, n, sigma=1.0, density=0.2):
    "Generates data matrix X and observations Y."
    np.random.seed(1)
    beta_star = np.random.randn(n)
    idxs = np.random.choice(range(n), int((1-density)*n), replace=False)
    for idx in idxs:
        beta_star[idx] = 0
    X = np.random.randn(m,n)
    Y = X.dot(beta_star) + np.random.normal(0, sigma, size=m)
    scaler = StandardScaler().fit(X)
    new_X = scaler.transform(X)
    #scale_y = Normalizer().fit(Y)
    #new_y = scaler.transform(Y)
    return new_X, Y, beta_star

def error_vs_time(n,d):
    # n = 10_000
    # d = 10
    # sklearn_regulariser = 10
    # sketch_size = 50*d
    # times2test = np.linspace(0.002, 0.01, 5)
    # results = {}
    #
    # for sketch_method in ihs_sketches:
    #     results[sketch_method] = {}
    #     for time in times2test:
    #         results[sketch_method][time] = {}
    #
    # print(results)
    #
    #
    # # Setup
    # # sparse_X, X, y, x_star = generate_lasso_data(n,d,data_density=1.0, sigma=1, sol_density=0.3, return_sparse=True)
    # X, y, x_star = my_lasso_data(n, d, sigma=1.0, density=0.25)
    # sparse_X = sparse.coo_matrix(X)
    # rows, cols, vals = sparse_X.row, sparse_X.col, sparse_X.data
    #
    # # Solve with SKLEARN for ground truth.
    # x_opt, f_opt, sklearn_time = sklearn_wrapper(X, y, n, d, regulariser=sklearn_regulariser, trials=1)
    # lsq_vs_truth_errors = prediction_error(X,x_star,x_opt) # nb need to log this for plots
    # ihs_lasso_bound = np.linalg.norm(x_opt,1)
    #
    # fig, ax = plt.subplots(figsize=(12,9))
    # for sketch_method in ihs_sketches:
    #     for time in times2test:
    #         ihs_lasso = ihs.IHS(data=X, targets=y, sketch_dimension=sketch_size,
    #                         sketch_type=sketch_method,
    #                         number_iterations=100,
    #                         data_rows=rows,data_cols=cols,data_vals=vals)
    #         print("STARTING IHS-LASSO ALGORITHM WITH {}".format(sketch_method), 60*"*")
    #         #start = default_timer()
    #         d = {'time_to_run' : time, 'problem' : "lasso", 'bound' : ihs_lasso_bound}
    #         #d = (0.1, 'lasso', ihs_lasso_bound)
    #         x_ihs, n_iters = ihs_lasso.solve_for_time(**d)
    #         soln_error = np.linalg.norm(X@(x_opt - x_ihs))**2/n #prediction_error(X,x_ihs,x_opt)
    #         results[sketch_method][time] = np.log(soln_error)
    #         print('{} iterations used'.format(n_iters))
    #         print('Solution Error: {}'.format(soln_error))
    #     ax.plot(times2test, results[sketch_method].values(), label=sketch_method)
    # print(results)
    # ax.legend()
    # ax.set_ylabel('Solution Error')
    # ax.set_xlabel('Time (s)')
    # plt.show()
    # pass
    '''Show that a random lasso instance is approximated by the
    hessian sketching scheme'''
    print(80*"-")
    print("TESTING LASSO ITERATIVE HESSIAN SKETCH ALGORITHM")

    n = 100000
    d = 15
    sketch_size = 5*d
    sklearn_lasso_bound = 5.0
    trials = 1
    lasso_time = 0
    times2test = np.linspace(0.005,0.4,40)
    print("Generating  data")
    X,y,x_star = my_lasso_data(n,d)
    X = normalize(X)
    print("Converting to COO format")
    sparse_data = sparse.coo_matrix(X)
    rows, cols, vals = sparse_data.row, sparse_data.col, sparse_data.data
    print("Beginning test")
    ### Test Sklearn implementation
    clf = Lasso(sklearn_lasso_bound)

    start = process_time()
    for i in range(trials):
        x_opt = sklearn_wrapper(X,y,n,d,sklearn_lasso_bound, trials)[0]
    end = process_time()
    sklearn_time = (end-start)/trials
    print("LASSO-skl time: {}".format(sklearn_time))

    time_results = {}
    for sketch in ihs_sketches:
        time_results[sketch] = {}

    fig, ax = plt.subplots()
    for sketch_method in ihs_sketches:
        ihs_lasso = ihs.IHS(data=X, targets=y, sketch_dimension=sketch_size,
                            sketch_type=sketch_method,
                            number_iterations=10,
                            data_rows=rows,data_cols=cols,data_vals=vals)
        sketch_distortions = []
        for time in times2test:
            print("STARTING IHS-LASSO ALGORITHM WITH {}".format(sketch_method), 60*"*")
            print("Testing time: {}".format(time))
            funct_dict = {'time_to_run' : time, 'problem' : "lasso", 'bound' : sklearn_lasso_bound}
            #x_ihs,_, _, _, iters_used = ihs_lasso.fast_solve({'problem' : "lasso", 'bound' : sklearn_lasso_bound},timing=True)
            x_ihs,iters_used = ihs_lasso.solve_for_time(**funct_dict)

            print("Comparing difference between opt and approx:")
            solution_error = np.log10((1/n)*np.linalg.norm(x_ihs-x_opt)**2) #prediction_error(X,x_opt,x_ihs)**2

            sketch_distortions.append(solution_error)
            print("||x^* - x'||_A^2: {}".format(np.log(solution_error)))
            print("{} iterations used".format(iters_used))
        time_results[sketch] = sketch_distortions
        max_val = np.max(sketch_distortions)
        yvals = np.array(sketch_distortions)
        ax.plot(times2test, yvals, label=sketch_method)
    ax.axvline(sklearn_time,color='k',label='Sklearn')
    ax.legend()
    #ax.set_ylabel("log(\|x-x'\|/n)")
    ax.set_xlabel('Time(seconds)')
    print(time_results)
    plt.show()



def main():
    error_vs_time(10000,200)



    pass

if __name__ == "__main__":
    main()
