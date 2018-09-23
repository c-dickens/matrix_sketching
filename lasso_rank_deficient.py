'''
Learning a LASSO with potentially losing rank
'''
import itertools
import numpy as np
from scipy import sparse
from sklearn.datasets import make_regression
from lib import countsketch, srht, gaussian, classical_sketch
from lib import ClassicalSketch
from lib import iterative_hessian_sketch as ihs
from synthetic_data_functions import generate_lasso_data
from utils import *
from experiment_parameter_grid import param_grid, sketch_names, sketch_functions
from experiment_parameter_grid import ihs_sketches as ihs_sketch_names
from matplotlib_config import update_rcParams
import matplotlib.pyplot as plt
from my_plot_styles import plotting_params

random_seed = param_grid['random_state']
np.random.seed(random_seed)

sketch_names.append("Exact")
sketch_names.append("Classical")

repeats = 1 #param_grid['num trials']

def experiment_error_vs_iteration():
    n = 60000
    d = 200
    sklearn_regulariser = 10
    gamma_vals = [5,7.5,10]
    ihs_sketch_names.append("Gaussian")
    number_iterations = [1,2,5,10] #np.asarray(np.linspace(2,10,5), dtype=np.int)

    # Output dictionaries
    error_to_opt = {sketch_name : {} for sketch_name in ihs_sketch_names}
    error_to_truth = {sketch_name : {} for sketch_name in ihs_sketch_names}
    for sketch_name in ihs_sketch_names:
        for gamma in gamma_vals:
            error_to_opt[sketch_name][gamma] = []
            error_to_truth[sketch_name][gamma] = []
    print(error_to_opt)
    print(error_to_truth)




    X, y, x_star = generate_lasso_data(n,d,data_density=0.4,\
                                 sigma=1.0, sol_density=0.2, return_sparse=False)
    #X,y = make_regression(n,d,d)
    sparse_X = sparse.coo_matrix(X)
    rows,cols,vals = sparse_X.row, sparse_X.col, sparse_X.data

    # Sklearn
    x_opt, f_opt, sklearn_time = sklearn_wrapper(X, y, n, d, regulariser=sklearn_regulariser, trials=1)
    #lsq_vs_truth_errors = prediction_error(X,x_star,x_opt) # nb need to log this for plots

    for gamma in gamma_vals:
        sketch_size = np.int(gamma*d)
        for ii in range(len(number_iterations)):
            print("-"*80)
            print("Testing gamma: {}, num_iterations: {}".format(gamma,number_iterations[ii]))
            iter_num = number_iterations[ii]
            for sketch_method in ihs_sketch_names:
                lsq_error, truth_error = 0,0
                for trial in range(repeats):
                    ihs_lasso = ihs.IHS(data=X, targets=y, sketch_dimension=sketch_size,
                                                            sketch_type=sketch_method,
                                                            number_iterations=number_iterations[ii],
                                                            data_rows=rows,data_cols=cols,data_vals=vals)
                    x_ihs,_,_,_,_,rank_fails = ihs_lasso.fast_solve({'problem' : "lasso", 'bound' : sklearn_regulariser}, timing=True,all_iterations=True,rank_check=True)
                    print("{}: ||x^* - x'||_A^2: {}".format(sketch_method, np.log10(np.linalg.norm(X@(x_opt - x_ihs)**2/X.shape[0]))))
                    print("{} rank failures ".format(rank_fails))
                    lsq_error += prediction_error(X,x_ihs, x_opt)**2
                    truth_error += prediction_error(X,x_ihs, x_star)**2
                    # print("Sketch: {}, opt_error: {}".format(sketch_method, lsq_error))
                mean_lsq_error = lsq_error/repeats
                mean_truth_error = truth_error/repeats
                error_to_opt[sketch_method][gamma].append(mean_lsq_error)
                error_to_truth[sketch_method][gamma].append(mean_truth_error)

    ### Save the dictionaries
    #error_to_opt_file = "figures/lasso_rank_deficient_error_opt_" + str(n) + ".npy"
    #error_to_truth_file = "figures/verify_ihs_error_to_truth" + str(n) + ".npy"
    #np.save(error_to_opt_file, error_to_opt)
    #np.save(error_to_opt_file, error_to_truth)
    ## Error plots
    # Plotting dict for gamma
    styles = ["--", "-", ":"]
    line_params = {gamma_vals[i] : styles[i] for i in range(len(styles))}
    print(error_to_opt)


    fig, ax = plt.subplots()
    for sketch_method in ihs_sketch_names:
        for gamma in gamma_vals:
            my_label = sketch_method + str(gamma)
            my_colour, my_marker = plotting_params[sketch_method]["colour"], plotting_params[sketch_method]["marker"]
            my_line = line_params[gamma]
            ax.plot(number_iterations, np.log(error_to_opt[sketch_method][gamma]),
             color=my_colour, marker=my_marker, linewidth=2, markersize=6,
             linestyle=my_line,label=my_label)

    ax.legend()
    ###### fig.savefig("figures/verify_ihs_error_to_opt.pdf", bbox_inches="tight")
    ax.set_title("Log Error to Sklearn")
    plt.show()

    fig, ax = plt.subplots()
    # Add a green line indicating the optimal estimate accuracy?
    for sketch_method in ihs_sketch_names:
        for gamma in gamma_vals:
            my_label = sketch_method + str(gamma)
            my_colour, my_marker = plotting_params[sketch_method]["colour"], plotting_params[sketch_method]["marker"]
            my_line = line_params[gamma]
            ax.plot(number_iterations, np.log(error_to_truth[sketch_method][gamma]),
             color=my_colour, marker=my_marker, linewidth=2, markersize=6,
             linestyle=my_line,label=my_label)
    ax.legend()
    ###### fig.savefig("figures/verify_ihs_error_to_truth.pdf", bbox_inches="tight")
    ax.set_title("Log Error to Truth")

    plt.show()

def main():
    experiment_error_vs_iteration()

if __name__ == "__main__":
    main()
