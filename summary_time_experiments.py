'''Experiment script for summary time

CWT is the Clarkson-Woodruff Transform (CountSketch) and SRHT is is the randomized
hadamard Transform'''
import json
import itertools
from pprint import PrettyPrinter
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
from experiment_parameter_grid import param_grid, sketch_names, sketch_functions

from matplotlib_config import update_rcParams
import matplotlib.pyplot as plt
from my_plot_styles import plotting_params
update_rcParams()



# # nb. the marker styles are for the plots with multiple sketch settings.
my_markers = ['.', 's', '^', 'D', 'x', '+', 'V', 'o', '*']
col_markers = {param_grid['columns'][i]: my_markers[i] for i in range(len(param_grid['columns']))}
# print(col_markers)
random_seed = param_grid['random_state']
np.random.seed(random_seed)
# sketch_names = ["CountSketch", "SRHT", "Gaussian"]
# sketch_functions = {"CountSketch": countsketch.CountSketch,
#                     "SRHT" : srht.SRHT,
#                     "Gaussian" : gaussian.GaussianSketch}

# param_grid = {
#         'num trials' : 5,
#         'rows' : [10000, 25000, 50000, 100000],#, 100000,250000],
#         'columns' : [10,50,100, 500, 1000],#, 100, 500, 1000],
#         'sketch_factors' : 5,
#         'density' : np.linspace(0.1,1.0, num=10)
#     }



# plotting_params = {"CountSketch" : {"colour" : "b",
#                                     "line_style" : '-',
#                                     "marker" : "o" },
#                    "SRHT" : {"colour" : "k",
#                              "marker" : "s",
#                              "line_style" : ':'},
#                    "Gaussian" : {"colour" : "r",
#                                  "marker" : "v",
#                                  "line_style" : "-."},
#                    "Classical" : {"colour" : "m",
#                                   "marker" : "*"},
#                     "Exact" : {"colour" : "mediumspringgreen",
#                                "marker" : "^"}
#                                   }


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

def full_summary_times_and_plots(row_list, column_list):
    '''This is a wrapper to do the above `experiment_summary_time_vs_sparsity`
    experiment over the entire parameter grid as defined above'''

    results = {
        "CountSketch" : {},
        "SRHT"        : {}
    }

    for key in results.keys():
        results[key] = {}
        for n in row_list:
            results[key][n] = {}
            for d in column_list:
                    results[key][n][d] = {}
    print(results)

    for n_rows in row_list:
        n_trials = param_grid['num trials']
        print('Testing design matrix: {} rows)'.format(n_rows))
        exp_result_list = Parallel(n_jobs=-1)(delayed(experiment_summary_time_vs_sparsity)\
                        (n_rows,cols,n_trials,5*cols,param_grid['density']) for cols in column_list)
        print(exp_result_list)

        for ii in range(len(column_list)):
            dicts = exp_result_list[ii]
            count_sketch_dict= dicts[0]
            srht_dict = dicts[1]
            results["CountSketch"][n_rows][column_list[ii]] = count_sketch_dict
            results["SRHT"][n_rows][column_list[ii]] = srht_dict

    print(results)

    file_name = 'figures/summary_time_vs_sparsity' + "_n_" + str(n) + '.npy'
    np.save(, results)
    print(json.dumps(results,indent=4))

    for n in param_grid['rows']:
        fig, ax = plt.subplots(dpi=250)
        for sketch,d in itertools.product(results.keys(), param_grid['columns']):
            print(n,sketch,d)
            my_colour = plotting_params[sketch]["colour"]
            my_label = sketch + str(d)
            my_line = plotting_params[sketch]["line_style"]
            # this just pulls the index of d in the param list and uses that as
            # the marker inded
            my_marker = my_markers[param_grid['columns'].index(d)]
            ax.plot(param_grid['density'], results[sketch][n][d].values(),
                    color=my_colour, linestyle=my_line, linewidth=2.0,
                    marker=my_marker, markersize=8.0,label=my_label)

        ax.legend(title='$n$ = {}'.format(n),loc=1)
        ax.set_yscale('log')
        ax.set_ylabel('log(seconds)')
        ax.set_xlabel('Density')
        save_name = "figures/summary_time_density_"+str(n)+".pdf"
        fig.savefig(save_name, bbox_inches="tight")
    plt.show()
    return results

def experiment_summary_distortion_vs_aspect_ratio(n_rows, n_trials, sketch_size=1.5,density=0.3):
    '''Experiment to see how distortion varies for a fixed sketch size over
    different aspect ratios.'''
    max_num_cols = np.int(n_rows/2)
    #cols = np.concatenate((np.asarray([10,100,250,500],dtype=np.int),np.linspace(1000,
    #                       max_num_cols,max_num_cols/1000,dtype=np.int)))
    cols = [10,50,75,100,250,500,750,1000,1500,2000,2500, 5000]
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
                print("Sketch: {}, testing {} cols with {} sketch_size, trial: {}".format(sketch, d,my_sketch_size, trial))
                summary = sketch_functions[sketch](data=A, sketch_dimension=my_sketch_size)
                S_A = summary.sketch(A)
                approx_norm = np.linalg.norm(S_A@x,ord=2)**2
                print("Approx ratio: {}".format(true_norm/approx_norm))
                print("Update val:{}".format(np.abs(approx_norm-true_norm) / true_norm))
                approx_factor += np.abs(approx_norm-true_norm)/true_norm
            distortions[sketch][d] = approx_factor/n_trials
    print(distortions)
    return distortions

def experiment_summary_time_distortion_real_data():

    # dataset = 'YearPredictionMSD'
    # data_path = 'data/'+ dataset + '.npy'

    all_datasets = datasets_config.datasets

    with open('figures/real_data_summary_time.json') as f:
        already_seen_data = json.load(f)
    print(already_seen_data.keys())

    saved_datasets = datasets_config.datasets.keys()
    new_data_2_sketch= {}

    for dataset in saved_datasets:
        if dataset not in already_seen_data.keys():
            # i.e we have added a new datasets to test
            new_data_2_sketch[dataset] = {}

    print(new_data_2_sketch)
    #new_exp_results = subspace_embedding_check(new_data_2_sketch)

    sketch_factors = [1,2,5]

    # Results dict
    results = {}


    for data in new_data_2_sketch:
        if data == "rucci":
            continue
        # if data is "kdd":
        #     continue
        # if data is "rucci":
        #     continue
        # if data is "rail2586":
        #     continue
        n_trials = all_datasets[data]["repeats"]
        print("-"*80)
        print("TESTING DATA {} WITH {} REPEATS".format(data, n_trials))

        #input_file = datasets[data]["filepath"]
        input_file = all_datasets[data]["filepath"]
        print(input_file)
        sparse_flag = False # a check to say if sparse data is found.
        # Defaults to False unles the sparse matrix can be read in.
        try:
            sparse_file = all_datasets[data]['filepath_sparse']
            sparse_flag = True
        except KeyError:
            print("No sparse representation")
        print(sparse_flag)

        if sparse_flag:
            print("Read in sparse format for {} as well as dense".format(data))
            sparse_data = load_npz(sparse_file)
            sX = sparse_data
            dense_data = sparse_data.toarray() #np.load(input_file)
            #X_row, X_col, X_data = sparse_data.row, sparse_data.col, sparse_data.data
        else:
            dense_data = np.load(input_file)

        X = dense_data[:,:]
        n,d = X.shape
        print("Dense shape of {}: {}".format(data, X.shape))
        # output dict structure
        results[data] = {"Exact Time" : 0}
        for sketch in ["CountSketch", "SRHT"]:
            results[data][sketch] = {}
            for factor in sketch_factors:
                results[data][sketch][factor] = {"sketch time"  : 0,
                                         "product time" : 0,
                                         "total time"   : 0,
                                         "error"        : 0}


        # True values
        print("-"*80)
        print("ENTERING EXPERIMENT LOOP FOR TRUE VALUES")
        cov_time_total = 0
        for _ in range(n_trials):
            if sparse_flag:
                sparse_type_covariance_matrix = sparse.csr_matrix((d,d))
                print("iteration ", _)
                cov_time_start = default_timer()
                sparse_type_covariance_matrix = sX.T@sX
                cov_time_total += default_timer() - cov_time_start
            else:
                print("iteration ", _)
                cov_time_start = default_timer()
                covariance_matrix = X.T@X
                cov_time_total += default_timer() - cov_time_start

        if sparse_flag:
            print("Covariance matrix type({})".format(type(sparse_type_covariance_matrix)))
            try:
                covariance_matrix_norm = np.linalg.norm(sparse_type_covariance_matrix, ord='fro')**2

            except ValueError:
                covariance_matrix = sparse_type_covariance_matrix.toarray() # if we had sparse representation then
                # need to convert covariance_matrix to ndarray
                covariance_matrix_norm = np.linalg.norm(covariance_matrix, ord='fro')**2
        else:
            covariance_matrix_norm = np.linalg.norm(covariance_matrix, ord='fro')**2

        cov_time_mean = cov_time_total/n_trials
        print("Exact time for {}: {}".format(data,cov_time_mean))
        results[data]["Exact Time"] = cov_time_mean
        true_rank = np.linalg.matrix_rank(X)



        print("True rank of matrix: {}".format(true_rank))
        print("-"*80)
        print("ENTERING EXPERIMENT LOOP FOR SKETCH")
        for sketch_method in ["CountSketch", "SRHT"]:
            if (sketch_method is "SRHT") and X.shape[1]>1000:
                print("CONTINUING AS DATA TOO LARGE FOR SRHT")
                continue
            for factor in sketch_factors:
                # Measurable variables
                summary_time = 0
                approx_hessian_time = 0
                distortion = 0
                rank_tests = np.zeros((n_trials,))
                print("TESTING {} with sketch size {}*d".format(sketch_method, factor))
                for _ in range(n_trials):
                    print("iteration ", _)
                    sketch_size = np.int(np.ceil(factor*d))
                    #summary = countsketch.CountSketch(data=X,sketch_dimension=sketch_sizes[0])
                    if sparse_flag is True and sketch_method is "CountSketch":
                        summary = sketch_functions[sketch_method](data=sX,sketch_dimension=sketch_size)
                    else:
                        summary = sketch_functions[sketch_method](data=X,sketch_dimension=sketch_size)
                    start = default_timer()
                    S_A = summary.sketch(data=X)
                    summary_time += default_timer() - start
                    sketch_rank = np.linalg.matrix_rank(S_A)
                    print("Sketch rank {}".format(sketch_rank))
                    if sketch_rank == true_rank:
                        rank_tests[_] = 1


                    hessian_start = default_timer()
                    approx_hessian = S_A.T@S_A
                    approx_hessian_time += default_timer() - hessian_start

                    distortion += np.linalg.norm(approx_hessian - covariance_matrix,
                                                ord='fro')**2/covariance_matrix_norm
                num_fails = n_trials - np.sum(rank_tests)
                mean_total_time = (approx_hessian_time  + summary_time)/n_trials
                mean_distortion = distortion / n_trials
                print("Mean total time {}".format(mean_total_time))
                print("Mean distortion: {}".format(mean_distortion))
                print("Number of rank fails: {}".format(num_fails))


                results[data][sketch_method][factor]["sketch time"] = summary_time/n_trials
                results[data][sketch_method][factor]["product time"] = approx_hessian_time/n_trials
                results[data][sketch_method][factor]["total time"] = mean_total_time
                results[data][sketch_method][factor]["error"] = mean_distortion
                results[data][sketch_method][factor]["rank fails"] = num_fails


    pretty = PrettyPrinter(indent=4)
    already_seen_data.update(results)
    pretty.pprint(already_seen_data)
    with open('figures/real_data_summary_time.json', 'w') as outfile:
       json.dump(already_seen_data, outfile)
    np.save("figures/real_data_summary_time.npy", already_seen_data)

    # np.save("figures/real_data_summary_time.npy", results)
    # with open('figures/real_data_summary_time.json', 'w') as outfile:
    #     json.dump(results, outfile)
    return already_seen_data

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
    summary_time_start = default_timer()
    full_summary_times_and_plots(param_grid['rows'], param_grid['columns'])
    summary_time_end = default_timer() - summary_time_start
    print("Script time: ", summary_time_end)


    ### COMPLETED EXPERIMENTS
    # distortions_to_plot = experiment_summary_distortion_vs_aspect_ratio(25000,param_grid['num trials'])
    # np.save("figures/distortion_vs_cols.npy", distortions_to_plot)
    # plotting_distortion(distortions_to_plot,25000,param_grid['num trials'])


     ### -----------

    # real_data_summary_time = experiment_summary_time_distortion_real_data()
    # print(json.dumps(real_data_summary_time,indent=4))
    # pass


if __name__ == "__main__":
    main()
