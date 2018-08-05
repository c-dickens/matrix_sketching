'''
Experimental script which verifies the IHS paper results with the count sketch,
SRHT, and Gaussian sketch.

Experiments:
1. MSE vs row dimension
'''
import itertools
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from lib import countsketch, srht, gaussian, classical_sketch
from lib import ClassicalSketch
from lib import iterative_hessian_sketch as ihs
from synthetic_data_functions import unconstrained_regession_data
import matplotlib.pyplot as plt

random_seed = 400
np.random.seed(random_seed)
sketch_names = ["Exact", "CountSketch", "SRHT","Gaussian", "Classical"]
ihs_sketch_names = ["CountSketch", "SRHT", "Gaussian"]
sketch_functions = {"CountSketch": countsketch.CountSketch,
                    "SRHT" : srht.SRHT,
                    "Gaussian" : gaussian.GaussianSketch,
                    "Classical" : classical_sketch.ClassicalSketch}
repeats = 10
plotting_params = {"CountSketch" : {"colour" : "b",
                                    "marker" : "o" },
                   "SRHT" : {"colour" : "k",
                             "marker" : "s"},
                   "Gaussian" : {"colour" : "r",
                                 "marker" : "v"},
                   "Classical" : {"colour" : "m",
                                  "marker" : "*"},
                    "Exact" : {"colour" : "mediumspringgreen",
                               "marker" : "^"}
                                  }


def mean_square_error(x1, x2):
    '''compute ||x2 - x1||_2^2'''
    return np.linalg.norm(x2-x1)**2

def prediction_error(data,x1,x2):
    '''compute np.sqrt(1/n)*||A(x1-x2)||_2'''
    return (1/np.sqrt(data.shape[0]))*np.linalg.norm(data@(x1-x2))






def experiment_mse_vs_row_dim():
    '''
    nb. need to square the prediction error'''
    row_dims = [100*2**i for i in range(3,15)]
    d = 10
    sketch_size = np.int(5*d)
    num_rounds = [1 + np.int(np.ceil(np.log2(n))) for n in row_dims]
    classical_sketch_size = [np.int(N*sketch_size) for N in num_rounds ]
    print("Using sketch size: {}".format(sketch_size))
    print("Classical sketch sizes: {}".format(classical_sketch_size))
    print("Num iterations: {}".format(num_rounds))

    # Output dictionaries
    MSE = {sketch_names[i] : np.zeros(len(row_dims),) for i in range(len(sketch_names)) }
    PRED_ERROR = {sketch_names[i] : np.zeros(len(row_dims),) for i in range(len(sketch_names)) }

    for n in row_dims:
        print("Testing {} rows".format(n))
        experiment_index = row_dims.index(n)
        for trial in range(repeats):
            X,y, x_true = unconstrained_regession_data(n,d,variance=1.0)
            print("TRIAL {}".format(trial))
            for sketch_method in sketch_names:
                if sketch_method is "Exact":
                    # Solve the initial regression
                    true_model = LinearRegression()
                    true_model.fit(X,y)
                    x_opt = true_model.coef_
                    MSE["Exact"][experiment_index] += mean_square_error(x_opt, x_true)
                    PRED_ERROR["Exact"][experiment_index] += prediction_error(X,x_opt, x_true)

                elif sketch_method is "Classical":
                    sketch_and_solve = ClassicalSketch(data=X, targets=y,
                                                        sketch_dimension=classical_sketch_size[experiment_index],
                                                        sketch_type="Gaussian")
                    x_classical = sketch_and_solve.solve()
                    MSE["Classical"][experiment_index] += mean_square_error(x_true, x_classical)
                    PRED_ERROR["Classical"][experiment_index] += prediction_error(X, x_true, x_classical)

                else:
                    ols_ihs = ihs.IHS(data=X, targets=y, sketch_dimension=sketch_size,
                                                            sketch_type=sketch_method,
                                                            number_iterations=num_rounds[experiment_index],
                                                            random_state=random_seed)
                    print("IHS ALGORITHM WITH {}".format(sketch_method))
                    #start = default_timer()
                    x_approx = ols_ihs.solve()
                    MSE[sketch_method][experiment_index] += mean_square_error(x_true, x_approx)
                    PRED_ERROR[sketch_method][experiment_index] += prediction_error(X, x_true, x_approx)

    print(MSE)
    print(PRED_ERROR)
    np.save("figures/verify_ihs_error_num_rows_mse.npy", MSE)
    np.save("figures/verify_ihs_error_num_rows_pred_error.npy", PRED_ERROR)

    fig, (ax0, ax1) = plt.subplots(1,2)
    for sketch_method in sketch_names:
        my_colour, my_marker = plotting_params[sketch_method]["colour"], plotting_params[sketch_method]["marker"]
        MSE[sketch_method] /= repeats
        PRED_ERROR[sketch_method] /= repeats
        if sketch_method is "Exact":
            my_label = "Optimal"
        else:
            my_label = sketch_method
        ax0.plot(row_dims, MSE[sketch_method], color=my_colour, marker=my_marker,
                                linewidth=2, markersize=6, label=my_label)
        ax1.plot(row_dims, PRED_ERROR[sketch_method], color=my_colour,
                    marker=my_marker,linewidth=2, markersize=6, label=my_label)

    ax0.set_xscale('log')
    ax0.set_yscale('log')
    ax0.set_xlabel("n")
    ax0.set_ylabel("Mean Square Error")
    ax0.legend()
    ax0.grid(True)

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True)
    ax1.set_xlabel("n")
    ax1.set_ylabel("Prediction Error")
    ax1.legend()
    plt.tight_layout()
    fig.savefig("figures/verify_ihs_error_num_rows.pdf", bbox_inches="tight")
    plt.show()

def experiment_error_vs_iteration():
    n = 6000
    d = 20
    gamma_vals = [4,6,8]
    ihs_sketch_names = ["CountSketch", "SRHT", "Gaussian"]
    number_iterations = np.asarray(np.linspace(2,40,20), dtype=np.int)

    # Output dictionaries
    error_to_lsq = {sketch_name : {} for sketch_name in ihs_sketch_names}
    error_to_truth = {sketch_name : {} for sketch_name in ihs_sketch_names}
    for sketch_name in ihs_sketch_names:
        for gamma in gamma_vals:
            error_to_lsq[sketch_name][gamma] = []
            error_to_truth[sketch_name][gamma] = []
    print(error_to_lsq)
    print(error_to_truth)

    X, y, x_star = make_regression(n_samples=n, n_features=d,\
                            n_informative=d,noise=1.0,coef=True)

    # Least squares estimator
    optimal = np.linalg.lstsq(X,y)
    x_ls = optimal[0]
    lsq_vs_truth_errors = np.log(1/np.sqrt(n))*np.linalg.norm(X@(x_ls-x_star))

    for gamma in gamma_vals:
        sketch_size = int(gamma*d)
        for ii in range(len(number_iterations)):
            print("Testing gamma: {}, num_iterations: {}".format(gamma,number_iterations[ii]))
            iter_num = number_iterations[ii]
            for sketch_method in ihs_sketch_names:
                lsq_error, truth_error = 0,0
                for trial in range(repeats):
                    print("{}, trial: {}".format(sketch_method, trial))
                    x_approx = ihs.IHS(data=X,targets=y,sketch_dimension=sketch_size,sketch_type=sketch_method,
                            number_iterations=iter_num, random_state=random_seed+ii).solve() # jsut putting + ii in to change the random seed
                    lsq_error += prediction_error(X,x_approx, x_ls)
                    truth_error += prediction_error(X,x_approx, x_star)
                mean_lsq_error = lsq_error/repeats
                mean_truth_error = truth_error/repeats
                error_to_lsq[sketch_method][gamma].append(mean_lsq_error)
                error_to_truth[sketch_method][gamma].append(mean_truth_error)

    ### Save the dictionaries
    np.save("figures/verify_ihs_error_to_lsq.npy", error_to_lsq)
    np.save("figures/verify_ihs_error_to_truth.npy", error_to_truth)
    ## Error plots
    # Plotting dict for gamma
    styles = ["--", "-", ":"]
    line_params = {gamma_vals[i] : styles[i] for i in range(len(styles))}



    fig, ax = plt.subplots()
    for sketch_method in ihs_sketch_names:
        for gamma in gamma_vals:
            my_label = sketch_method + str(gamma)
            my_colour, my_marker = plotting_params[sketch_method]["colour"], plotting_params[sketch_method]["marker"]
            my_line = line_params[gamma]
            ax.plot(number_iterations, np.log(error_to_lsq[sketch_method][gamma]),
             color=my_colour, marker=my_marker, linewidth=2, markersize=6,
             linestyle=my_line,label=my_label)

    ax.legend()
    fig.savefig("figures/verify_ihs_error_to_lsq.pdf", bbox_inches="tight")
    ax.set_title("Log Error to LSQ")

    fig, ax = plt.subplots()
    # Add a green line indicating the optimal estimate accuracy?
    for sketch_method in ihs_sketch_names:
        for gamma in gamma_vals:
            my_label = sketch_method + str(gamma)
            my_colour, my_marker = plotting_params[sketch_method]["colour"], plotting_params[sketch_method]["marker"]
            my_line = line_params[gamma]
            ax.plot(number_iterations, 1+np.log(error_to_truth[sketch_method][gamma]),
             color=my_colour, marker=my_marker, linewidth=2, markersize=6,
             linestyle=my_line,label=my_label)
    ax.legend()
    fig.savefig("figures/verify_ihs_error_to_truth.pdf", bbox_inches="tight")
    ax.set_title("Log Error to Truth")

    plt.show()

def experiment_error_vs_dimensionality():
    dimension = [2**i for i in range(4,9)]


    # Output dictionaries
    error_to_truth = {sketch_name : {} for sketch_name in sketch_names}
    for sketch_name in sketch_names:
        for d in dimension:
            error_to_truth[sketch_name][d] = 0
    print(error_to_truth)

    for d in dimension:
        n = 250*d
        ii = dimension.index(d)
        sampling_rate = 10
        num_iterations = 1+np.int(np.log(n))
        for trial in range(5):
            # Generate the data
            X, y, x_star = make_regression(n_samples=n, n_features=d,\
                                    n_informative=d,noise=1.0,coef=True)
            for sketch in sketch_names:
                print("TRIAL {}: Testing {} on {}".format(trial, sketch,d))
                if sketch is "Exact":
                    x = np.linalg.lstsq(X,y)[0]
                    opt_error = prediction_error(X, x_star, x)
                elif sketch is "Classical":
                    sketch_size = sampling_rate*num_iterations*d
                    #sketch_size = sampling_rate*d
                    print("Classic sketch with {} sketch size".format(sketch_size))
                    x = ClassicalSketch(data=X, targets=y,
                                        sketch_dimension=sketch_size,
                                        sketch_type="SRHT").solve()
                else:
                    sketch_size = sampling_rate*d
                    print("Using {} iterations, sketch_size {}".format(num_iterations, sketch_size))
                    x = ihs.IHS(data=X,targets=y,sketch_dimension=sketch_size,
                                sketch_type=sketch, number_iterations=num_iterations,
                                random_state=random_seed).solve()

                opt_error = prediction_error(X,x_star,x)
                error_to_truth[sketch][d] += opt_error

    print(error_to_truth)
    ### Save the dictionaries
    np.save("figures/verify_ihs_error_dimension.npy", error_to_truth)
    for sketch in sketch_names:
        for d in dimension:
            error_to_truth[sketch][d] /= repeats

    print(error_to_truth)
    # plotting tools
    index = range(len(dimension))
    bar_width = 0.15
    fig, ax = plt.subplots()

    for ii in index:
        d = dimension[ii]
        exact_rects = ax.bar(index[ii], error_to_truth["Exact"][d], bar_width,color=plotting_params['Exact']["colour"])
        classical_rects = ax.bar(index[ii]+bar_width, error_to_truth["Classical"][d], bar_width,color=plotting_params['Classical']["colour"])
        countsketch_rects=classical_rects = ax.bar(index[ii]+2*bar_width, error_to_truth["CountSketch"][d], bar_width,color=plotting_params['CountSketch']["colour"])
        srht_rects=classical_rects = ax.bar(index[ii]+3*bar_width, error_to_truth["SRHT"][d], bar_width,color=plotting_params['SRHT']["colour"],label="SRHT")
        gaussian_rects=classical_rects = ax.bar(index[ii]+4*bar_width, error_to_truth["Gaussian"][d], bar_width,color=plotting_params['Gaussian']["colour"])
    ax.set_xticks(np.asarray(index,dtype=np.float) + 2*bar_width)
    ax.set_xticklabels(dimension)
    #ax.legend()
    fig.savefig("figures/verify_ihs_error_dimension.pdf", bbox_inches="tight")
    plt.show()


def main():
    experiment_mse_vs_row_dim()
    #experiment_error_vs_iteration()
    #experiment_error_vs_dimensionality()

if __name__ == "__main__":
    main()
