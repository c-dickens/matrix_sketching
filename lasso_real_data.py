'''
Real dataset LASSO experiments
'''
import json
import itertools
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
from experiment_parameter_grid import param_grid


def lasso_real_data():

    # dataset = 'YearPredictionMSD'
    # data_path = 'data/'+ dataset + '.npy'

    datasets = datasets_config.datasets
    sketch_factors = [200]

    # Results dict
    results = {}
    sklearn_lasso_bound = 10


    for data in datasets:
        subset_size = 200000

        if data is "kdd":
            continue
        if data is "rucci":
            continue
        if data is "rail2586":
            continue
        # if data is "slice":
        #     continue
        #if data is "susy":
        #    continue
        # if data is "YearPredictionMSD":
        #     continue


        n_trials = datasets[data]["repeats"]
        print("-"*80)
        print("TESTING DATA {} WITH {} REPEATS".format(data, n_trials))

        #input_file = datasets[data]["filepath"]
        input_file = datasets[data]["filepath"]
        sparse_flag = False # a check to say if sparse data is found.
        # Defaults to False unles the sparse matrix can be read in.
        try:
            sparse_file = datasets[data]['filepath_sparse']
            sparse_flag = True
        except KeyError:
            print("No sparse representation")
        print("SPARSE FLAG: {}".format(sparse_flag))

        if sparse_flag:
            print("Read in sparse format for {} as well as dense".format(data))
            sparse_data = load_npz(sparse_file).tocsr()
            if data is "census":
                cols2keep = [i for i in range(sparse_data.shape[1]) if i != 1]
                sparse_data = sparse_data[:,cols2keep]
            print(sparse_data.shape)
            sparseX = coo_matrix(sparse_data[:subset_size,:-1])
            sparsey = sparse_data[:subset_size,-1]
            scaler = StandardScaler(with_mean=False)
            sX = scaler.fit_transform(sparseX)
            sX = coo_matrix(sX)
            print(sX.shape)
            print(type(sX))
            dense_data = np.load(input_file)
            X_row, X_col, X_data = sX.row, sX.col, sX.data
        else:
            dense_data = np.load(input_file)


        if data is "census":
            cols2keep = [i for i in range(dense_data.shape[1]) if i != 1]
        else:
            cols2keep = [i for i in range(dense_data.shape[1])]

        if data is "california_housing_train":
            subset_size = dense_data.shape[0]

        dX, y = dense_data[:subset_size,cols2keep], dense_data[:subset_size,-1]
        scaler = StandardScaler()
        X = scaler.fit_transform(dX)
        n,d = X.shape
        print("Dense shape of {}: {}".format(data, X.shape))

        # output dict structure
        results[data] = {"sklearn" : {"solve time" : 0,
                                      #"x_opt"      : np.zeros((d,)),
                                       "objective value"  : 0}}
        for sketch in ["CountSketch", "SRHT"]:
            results[data][sketch] = {}
            for factor in sketch_factors:
                results[data][sketch][factor] = {"error to sklearn"   : 0, #np.linalg.norm(X@(x0-x_opt),ord=2)**2,
                                     "error to truth"     : 0, #np.linalg.norm(X@(x0-truth),ord=2)**2,
                                     "objective val"      : 0, #original_lasso_objective(X,y,sklearn_lasso_bound,x0),
                                     "setup time"         : 0, #setup_time,
                                     "sketch_time"        : 0, #sketch_time,
                                     "optimisation time"  : 0, #opt_time,
                                     "total time"         : 0, #setup_time+sketch_time+opt_time,
                                     "num iters"          : 0, #n_iters,
                                     "num columns"        : d}



        # True values
        print("LASSO-SKLEARN EXPERIMENT LOOP")
        # if sparse_flag is True:
        #     x_opt, f_top, sklearn_mean_time = sklearn_wrapper(sX,sy,n,d, sklearn_lasso_bound, n_trials)
        # else:
        x_opt, f_opt, sklearn_mean_time = sklearn_wrapper(X,y,n,d, sklearn_lasso_bound, n_trials)
        print("Time to train LASSO on {} data: {}".format(data, sklearn_mean_time))
        results[data]['sklearn']["solve time"] = sklearn_mean_time
        results[data]['sklearn']['objective value'] = f_opt

        print("ENTERING EXPERIMENT LOOP FOR SKETCH")

        for sketch_method in ["CountSketch", "SRHT"]:
            if (sketch_method is "SRHT") and X.shape[1]>500:
                print("CONTINUING AS DATA TOO LARGE FOR SRHT")
                continue
    #
            for factor in sketch_factors:
                # Measurable variables
                summary_time = 0
                approx_hessian_time = 0
                distortion = 0
                print("TESTING {} with sketch size {}*d".format(sketch_method, factor))
                for _ in range(1):
                    print("Trial number {}".format(_+1))
                    sketch_size = np.int(np.ceil(factor*d))


                    if sparse_flag is True and sketch_method is "CountSketch":
                        ihs_lasso = IHS(data=X, targets=y, sketch_dimension=sketch_size,
                                        sketch_type=sketch_method,number_iterations=20,#1+np.int(np.ceil(np.log(n))),
                                        data_rows=X_row,data_cols=X_col,data_vals=X_data,
                                        random_state=param_grid["random_state"])

                    else:
                        ihs_lasso = IHS(data=X, targets=y, sketch_dimension=sketch_size,
                                        sketch_type=sketch_method,number_iterations=20,#1+np.int(np.ceil(np.log(n))),
                                        random_state=param_grid["random_state"])

                    x0, setup_time, sketch_time, opt_time, n_iters = ihs_lasso.fast_solve({'problem' : "lasso", 'bound' : sklearn_lasso_bound}, timing=True)
                    #x0 /= n
                    print("IHS {} time {}".format(sketch_method, setup_time + sketch_time + opt_time))

                # Need an averagin term in here.
                results[data][sketch_method][factor] = {#"estimate"           : x0,
                                      "error to sklearn"   : np.linalg.norm(x0-x_opt,ord=2)**2,
                                      "prediction error"   : (1/n)*np.linalg.norm(X@(x0-x_opt),ord=2)**2,
                                      "objective val"      : original_lasso_objective(X,y,sklearn_lasso_bound,x0),
                                      "setup time"         : setup_time,
                                      "sketch_time"        : sketch_time,
                                      "optimisation time"  : opt_time,
                                      "total time"         : setup_time+sketch_time+opt_time,
                                      "num iters"          : n_iters,
                                      "num columns"        : d}
                print(np.c_[x0[:,None], x_opt[:,None]])
                #print(x_opt)
                # results[data][sketch_method][factor]["sketch time"] = summary_time/n_trials
                # results[data][sketch_method][factor]["product time"] = approx_hessian_time/n_trials
                # results[data][sketch_method][factor]["total time"] = mean_total_time
                # results[data][sketch_method][factor]["error"] = mean_distortion
    np.save("figures/real_data_lasso.npy", results)
    print(json.dumps(results,indent=4))
    #with open('figures/real_data_lasso.json', 'w') as outfile:
    #    json.dump(results, outfile)
    return results

if __name__ == "__main__":
    lasso_real_data()
