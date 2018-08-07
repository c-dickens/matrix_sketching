'''Experiment script for summary time

CWT is the Clarkson-Woodruff Transform (CountSketch) and SRHT is is the randomized
hadamard Transform'''
import itertools
import numpy as np
import scipy as sp
from scipy import sparse
from timeit import default_timer
from lib import countsketch, srht, gaussian, classical_sketch
from lib import ClassicalSketch


import matplotlib.pyplot as plt

random_seed = 400
np.random.seed(random_seed)
sketch_names = ["CountSketch", "SRHT", "Gaussian"]
sketch_functions = {"CountSketch": countsketch.CountSketch,
                    "SRHT" : srht.SRHT,
                    "Gaussian" : gaussian.GaussianSketch}

param_grid = {
        'num trials' : 5,
        'rows' : [1000],
        'columns' : [10, 50],#, 500, 1000],
        'density' : np.linspace(0.01,0.1, num=10)
    }

# nb. the marker styles are for the plots with multiple sketch settings.
my_markers = ['.', 's', '^', 'v', 's', 'D', 'x', '+', 'o', '*']
col_markers = {param_grid['columns'][i]: my_markers[i] for i in range(len(param_grid['columns']))}
print(col_markers)

plotting_params = {"CountSketch" : {"colour" : "b",
                                    "line_style" : '-',
                                    "marker" : "o" },
                   "SRHT" : {"colour" : "k",
                             "marker" : "s",
                             "line_style" : ':'},
                   "Gaussian" : {"colour" : "r",
                                 "marker" : "v",
                                 "line_style" : "-."},
                   "Classical" : {"colour" : "m",
                                  "marker" : "*"},
                    "Exact" : {"colour" : "mediumspringgreen",
                               "marker" : "^"}
                                  }



def experiment_summary_time_vs_sparsity(n_rows, n_cols, n_trials, sketch_size, densities):
    '''
    Function to automate the experiments measuring time
    and varying sparsity for given input data.

    Inputs: n_rows, n_cols, n_trials, sketch_size all int
            densities - list of densities to try

    Output: Two dictionaries with key as density value tested and
    then the times associated with testing that experiment.
    One dict for CWT and one for SRHT.
    '''
    rho_str = [str(density) for density in densities]
    CWT_time = {}
    SRHT_time = {}


    for density in densities:
        A = sparse.random(n_rows,n_cols,density).toarray()
        print('Testing density', density)
        for trial in range(n_trials):
            print("Trial no. ", trial)
            CWT_summary_time = 0
            SRHT_summary_time = 0


            CWT_summary = countsketch.CountSketch(data=A, sketch_dimension=sketch_size)
            print("Timing CountSketch")
            start = default_timer()
            CWT_A = CWT_summary.sketch(A)
            CWT_summary_time += default_timer() - start
            print("Timing SRHT")
            SRHT_summary = srht.SRHT(data=A, sketch_dimension=sketch_size)
            start = default_timer()
            SRHT_A = SRHT_summary.sketch(A)
            SRHT_summary_time = default_timer() - start

        CWT_time[density] = CWT_summary_time/n_trials
        SRHT_time[density] = SRHT_summary_time/n_trials

    return CWT_time, SRHT_time

def experiment_summary_distortion_vs_aspect_ratio(n_rows, n_trials, sketch_size=1.5,density=0.3):
    '''Experiment to see how distortion varies for a fixed sketch size over
    different aspect ratios.'''
    max_num_cols = np.int(n_rows/2)
    cols = np.concatenate((np.asarray([10,100,250,500],dtype=np.int),np.linspace(1000,
                           max_num_cols,max_num_cols/1000,dtype=np.int)))
    #cols = [10,50,75,100,250,500,750,1000,1500,2000,2500]
    print(cols)
    # output dicts
    distortions = {sketch : {} for sketch in sketch_functions.keys()}


    print("Entering loop")
    for d in cols:
        A = sparse.random(n_rows,d,density).toarray()#np.random.randn(n_rows,d)
        x = np.random.randn(d)
        x = x/np.linalg.norm(x)
        true_norm = np.linalg.norm(A@x,ord=2)**2
        my_sketch_size = np.int(sketch_size*d)
        if my_sketch_size >= n_rows:
            continue
        print(my_sketch_size)
        for sketch in sketch_functions.keys():
            #if sketch is "Gaussian":
            #    continue
            approx_factor = 0
            for trial in range(n_trials):
                print("Testing {} with {},sketch_size {}, trial: {}".format(d,sketch,my_sketch_size, trial))
                summary = sketch_functions[sketch](data=A, sketch_dimension=my_sketch_size)
                S_A = summary.sketch(A)
                approx_norm = np.linalg.norm(S_A@x,ord=2)**2
                print("Approx ratio: {}".format(true_norm/approx_norm))
                print("Update val:{}".format(np.abs(approx_norm-true_norm) / true_norm))
                approx_factor += np.abs(approx_norm-true_norm)/true_norm
            distortions[sketch][d] = approx_factor/n_trials
    print(distortions)
    return distortions

def experiment_summary_time_distortion_real_data(n_trials):

    dataset = 'YearPredictionMSD'
    data_path = 'data/'+ dataset + '.npy'
    rawdata_mat = np.load(data_path)

    #X = rawdata_mat[:, 1:]
    #data
    X = sparse.random(1000000,500, 0.2)
    print(type(X))
    #X = data.toarray()
    y = rawdata_mat[:, 0]
    y = y[:,None]

    print("Shape of data: {}".format(rawdata_mat.shape))
    print("Shape of testing data: {}".format(X.shape))
    print("Shape of test vector: {}".format(y.shape))

    # True quantities
    cov_time_total = 0
    for _ in range(n_trials):
        print("iteration ", _)
        cov_time_start = default_timer()
        covariance_matrix = X.T@X
        cov_time_total += default_timer() - cov_time_start
    cov_time_mean = cov_time_total/n_trials
    print("Exact time: {}".format(cov_time_mean))

    # True values
    # covariance_matrix_norm = np.linalg.norm(covariance_matrix, ord='fro')
    # XTy_mat = X.T@y
    # XTy_mat_norm = np.linalg.norm(XTy_mat, ord='fro')



    # Approximate Hessian
    summary_time = 0
    approx_hessian_time = 0
    for _ in range(n_trials):
        print("iteration ", _)
        summary = countsketch.CountSketch(data=X,sketch_dimension=600)
        start = default_timer()
        S_A = summary.sketch(data=X)
        summary_time += default_timer() - start

        hessian_start = default_timer()
        approx_hessian = S_A.T@S_A
        approx_hessian_time += default_timer() - hessian_start

    mean_total_time = approx_hessian_time  + summary_time

    print("Total time {}".format(mean_total_time))








### Plotting functions:

def plotting_summary_time_vs_sparsity(exp_results):
    fig, ax = plt.subplots(dpi=250, facecolor='w', edgecolor='k')
    #fig.suptitle("Sketch time vs Data density")

    for parameters in exp_results.keys():
        print(parameters)
        n = int(parameters[0])
        d = int(parameters[1])
        method = parameters[2]

        # Maintain consistency with other plots
        # if method == 'CountSketch':
        #     col = "b"
        #     line_style = '-'
        # else:
        #     col = "k"
        #     line_style = ':'
        my_colour = plotting_params[method]["colour"]
        my_line_style = plotting_params[method]["line_style"]
        my_label = method + str(d)
        ax.plot(*zip(*sorted(exp_results[parameters].items())),
                color=my_colour, marker=col_markers[d],linestyle=line_style,
                label=my_label)



    ax.set_yscale('log')
    ax.set_xlabel("Density")
    ax.set_ylabel("Time (seconds)")
    ax.legend()
    #ax.grid()
    #lgd = ax.legend(handles, labels, bbox_to_anchor=(1.0, 1.0))
    # fig.savefig('sparsity-time-1e5.pdf', #bbox_extra_artists=(lgd,),
    #         bbox_inches='tight',orientation='landscape')
    plt.show()


def plotting_distortion(distortion_results,n_rows=None,sketch_size=None):
    #pass

    fig, ax = plt.subplots(dpi=250, facecolor='w', edgecolor='k')

    for name in distortion_results.keys():
        sketch_method = name
        print(sketch_method)
        cols = distortion_results[sketch_method].keys()
        distortions = distortion_results[sketch_method].values()
        ax.plot(cols, distortions,
                color=plotting_params[sketch_method]["colour"],
                marker=plotting_params[sketch_method]["marker"],
                linestyle=plotting_params[sketch_method]["line_style"],
                label=sketch_method)

    ax.set_ylabel("Distortion")
    ax.set_xlabel("Number of columns")
    ax.set_yscale("log")
    if sketch_size is None or n_rows is None:
        ax.legend()
    else:
        ax.legend(title='(n,m) = ({},{}d)'.format(n_rows,sketch_size))
    fig.tight_layout()
    fig.savefig("figures/distortion_vs_cols.pdf", bbox_inches="tight")
    plt.show()


def main():
    ######### Script setup parameters ############
    # NB. Read from file in future

    # results = {}
    #
    # for n_rows, n_cols in itertools.product(param_grid['rows'], param_grid['columns']):
    #     n_trials = param_grid['num trials']
    #     print('Testing design matrix ({},{})'.format(n_rows, n_cols))
    #     sketch_size = 10*n_cols
    #     C_T, S_T =  experiment_summary_time_vs_sparsity(n_rows, n_cols, n_trials,sketch_size,param_grid['density'])
    #     results[(n_rows, n_cols, "CountSketch")] = C_T
    #     results[(n_rows, n_cols, "SRHT")] = S_T
    # print(results)

    #plotting_summary_time_vs_sparsity(results)
    # A = np.random.randn(10000, 50)
    # sum = countsketch.CountSketch(A, 500).sketch(A)
    # x = np.random.randn(50,)
    # x = x/np.linalg.norm(x)
    # true_norm = np.linalg.norm(A@x)
    # approx_norm = np.linalg.norm(sum@x)
    # print(true_norm**2)
    # print(approx_norm**2)
    # print(np.abs(true_norm**2 - approx_norm**2)/approx_norm**2)

    # distortions_to_plot = experiment_summary_distortion_vs_aspect_ratio(10000,20)
    # np.save("figures/distortion_vs_cols.npy", distortions_to_plot)
    # plotting_distortion(distortions_to_plot,10000,1.5)

    experiment_summary_time_distortion_real_data(5)

if __name__ == "__main__":
    main()