from timeit import default_timer
import numpy as np
import pandas as pd
from scipy.sparse import random
import matplotlib.pyplot as plt
import itertools
#from joblib import Parallel, delayed
import multiprocessing
import quadprog as qp

# Methods to test
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy import optimize
from scipy.sparse import coo_matrix
#import cvxpy as cp
from line_profiler import LineProfiler
np.random.seed(10)
import numba
from numba import jit

### SET NUMBER OF CORES
# N_JOBS = multiprocessing.cpu_count()
# print("Using {} jobs".format(N_JOBS))

### Helper functions

def encode2sparse(data):
    '''Given a data in the form of a ndarray return the dictionary which
    defines its sparsity pattern.'''
    X = data
    X_sparse = coo_matrix(X)
    X_rows = X_sparse.row
    X_cols = X_sparse.col
    X_data = X_sparse.data

    data_list = [ [] for i in range(X.shape[0])]
    current_row = 0
    counter = 0

    for row_id in X_rows:

        col,val = X_cols[counter], X_data[counter]
        if row_id == current_row:

            data_list[current_row].append( (col, val) )
        else:

            current_row = row_id
            data_list[current_row].append( (col, val) )
        counter += 1

    return data_list

def decode(data_list):
    '''Given a dictionary which defines the sparsity pattern of the data
    decode into [row, column, data] arrays.'''
    # Unpack the dictionary

    nrows, ncols, ndata = [], [], []
    for item_id in range(len(data_list)):
        for pair in data_list[item_id]:
            col,val = pair[0], pair[1]
            nrows.append(item_id)
            ncols.append(col)
            ndata.append(val)
    #print([nrows, ncols, ndata])

    nrows = np.asarray(nrows, dtype=np.int)
    ncols = np.asarray(ncols, dtype=np.int)
    ndata = np.asarray(ndata)
    return nrows, ncols, ndata

def check_encode_decode(data):
    '''Given data, encodes the data to a dict then decodes and checks the
    arrays match those of the arrays generated from the original sparse
    representation of the data.'''
    data_dict = encode2sparse(data)
    rows, cols, vals = decode(data_dict)
    X_sparse = coo_matrix(data)
    X_rows = X_sparse.row
    X_cols = X_sparse.col
    X_data = X_sparse.data

    print("Rows correct? {}".format(np.all(rows == X_rows)))
    print("Cols correct? {}".format(np.all(cols == X_cols)))
    print("Data correct? {}".format(np.all(vals == X_data)))



### QP approaches
def consQP_lasso(data, targets, regulariser):
    d = data.shape[1]
    Q = data.T@data
    big_hessian = np.vstack((np.c_[Q, -Q], np.c_[-Q,Q])) + 1E-10*np.eye(2*d)
    big_linear_term = np.hstack((-targets.T@data, targets.T@data))

    I_d = np.eye(d)
    constraint_matrix = np.vstack((np.eye(2*d), np.c_[I_d, I_d]))
    constraint_vals = np.zeros((3*d))
    constraint_vals[:d] = regulariser

    result = qp.solve_qp(big_hessian, big_linear_term, -constraint_matrix.T, constraint_vals)
    print(result)
    print(result[0])

    return result

def consQP_lasso_ihs(sketch, ATy, data, regulariser, old_x):
    d = data.shape[1]
    Q = data.T@data
    big_hessian = np.vstack((np.c_[Q, -Q], np.c_[-Q,Q])) + 1E-10*np.eye(2*d)
    linear_term = ATy - data.T@(data@old_x)
    big_linear_term = np.hstack((-linear_term, linear_term))

    I_d = np.eye(d)
    constraint_matrix = np.vstack((np.eye(2*d), np.c_[I_d, I_d]))
    constraint_vals = np.zeros((3*d))
    constraint_vals[:d] = regulariser

    result = qp.solve_qp(big_hessian, big_linear_term, -constraint_matrix.T, constraint_vals)
    #print(result)
    #print(result[0])

    return result

def ihs_constrained_qp(data, X_rows, X_cols, X_data, targets, regulariser, sketch_size, max_iters):
    '''
    Original problem is min 0.5*||Ax-b||_2^2 + lambda||x||_1
    IHS asks us to minimise 0.5*||SAx||_2^2 - <A^Tb, x> over
    the constraints.

    '''
    setup_time_start = default_timer()
    A = data

    y = targets
    _lambda = regulariser
    n,d = A.shape
    x0 = np.zeros(shape=(d,))
    m = int(sketch_size) # sketching dimension

    ATy = A.T@y

    #covariance_mat = A.T@A
    setup_time = default_timer() - setup_time_start
    old_norm = 1.0
    norm_diff = 1.0
    n_iter = 0

    opt_time = 0
    sketch_time = 0

    for n_iter in range(max_iters):
        if norm_diff > 1E-5:
            print("ITERATION {}".format(n_iter))

            start_sketch_time = default_timer()
            #S_A = _countSketch(A, sketch_size)
            S_A = _countSketch_fast(X_rows, X_cols, X_data, n, d, sketch_size) #
            end_sketch_time = default_timer() - start_sketch_time
            #print("SKETCH TIME: {}".format(end_sketch_time))
            sketch_time += end_sketch_time


            start = default_timer()
            #sub_prob = lasso_qp_iters(S_A, ATy, covariance_mat, _lambda, x0)
            sub_prob = consQP_lasso_ihs(S_A, ATy, A, _lambda, x0)
            end_opt_time = default_timer()-start
            opt_time += end_opt_time
            #print("OPT TIME: {}".format(end_opt_time))
            #print(sub_prob)

            x_out = sub_prob[2]
            x_new = x_out[d:] - x_out[:d]
            norm_diff = np.linalg.norm(x_new - x0)**2/old_norm

            #print(norm_diff)
            x0 += x_new
            old_norm = np.linalg.norm(x0)**2
            #n_iter += 1
        else:
            break
    print("Setup cost: {}".format(setup_time))
    print("Total sketching time: {}".format(sketch_time))
    print("Total optimization time: {}".format(opt_time))

    return x0, sketch_time, opt_time






def qp_lasso(data, targets, _lambda):
    '''solve lasso qp in quadprog in the form

    0.5*x^T (A^T A) x - (A^Tb + np.ones(d)).T@x

    Returns:

    result[0] - is the optimal solution'''
    A = data
    b = targets
    linear_term = (targets.T@A + _lambda*np.ones(A.shape[1])).T # to maintain consistency with github page

    result = qp.solve_qp(A.T@A, linear_term)
    return result
def lasso_qp_iters(sketch, ATy, data, _lambda, old_x):
#def lasso_qp_iters(sketch, ATy, covariance_mat, _lambda, old_x):
    '''
    solvers the iteration subproblem:
    min 0.5*norm(SA u)**2 - (b - Axt + _lambda*ones)^TAu

    Input - sketch -- SA
          -

    Output - result of the optimization
    '''
    linear_term = ATy - data.T@(data@old_x) + _lambda*np.ones(sketch.shape[1])
    result = qp.solve_qp(sketch.T@sketch, linear_term)
    return result


def ihs_lasso_qp(data, X_rows, X_cols, X_data, targets, regulariser, sketch_size, max_iters):
    '''
    Original problem is min 0.5*||Ax-b||_2^2 + lambda||x||_1
    IHS asks us to minimise 0.5*||SAx||_2^2 - <A^Tb, x> over
    the constraints.

    '''
    setup_time_start = default_timer()
    A = data

    y = targets
    _lambda = regulariser
    n,d = A.shape
    x0 = np.zeros(shape=(d,))
    m = int(sketch_size) # sketching dimension

    ATy = A.T@y

    #covariance_mat = A.T@A
    setup_time = default_timer() - setup_time_start
    old_norm = 1.0
    norm_diff = 1.0
    n_iter = 0

    opt_time = 0
    sketch_time = 0

    for n_iter in range(max_iters):
        if norm_diff > 1E-3:
            print("ITERATION {}".format(n_iter))

            start_sketch_time = default_timer()
            #S_A = _countSketch(A, sketch_size)
            S_A = _countSketch_fast(X_rows, X_cols, X_data, n, d, sketch_size) #
            end_sketch_time = default_timer() - start_sketch_time
            #print("SKETCH TIME: {}".format(end_sketch_time))
            sketch_time += end_sketch_time


            start = default_timer()
            #sub_prob = lasso_qp_iters(S_A, ATy, covariance_mat, _lambda, x0)
            sub_prob = lasso_qp_iters(S_A, ATy, A, _lambda, x0)
            end_opt_time = default_timer()-start
            opt_time += end_opt_time
            #print("OPT TIME: {}".format(end_opt_time))
            #print(sub_prob)

            x_new = sub_prob[0]

            norm_diff = np.linalg.norm(x_new - x0)**2/old_norm

            #print(norm_diff)
            x0 += x_new
            old_norm = np.linalg.norm(x0)**2
            n_iter += 1
        else:
            break
    print("Setup cost: {}".format(setup_time))
    print("Total sketching time: {}".format(sketch_time))
    print("Total optimization time: {}".format(opt_time))

    return x0, sketch_time, opt_time

### Baseline lasso functions in scipy ###
def lasso_scipy(x, data, targets, _lambda):
    return np.linalg.norm(data@x - targets,ord=2)**2 + _lambda*np.linalg.norm(x,1)

def scipy_lasso(data, targets, _lambda):
    result = optimize.minimize(lasso_scipy,np.zeros(data.shape[1]),
                              args=(data, targets, _lambda),
                              method='SLSQP',
                              tol=1E-10)
    return result

@jit('float64[:,:](float64[:,:], int64)', nopython=True)
def countsketch(a_mat, s_int):
    '''
    Count Sketch for Dense Matrix

    Input
        a_mat: m-by-n dense matrix A;
        s_int: sketch size.
    Output
        sketch_a_mat: m-by-s matrix A * S.
        Here S is n-by-s sketching matrix.
    '''

    m_int, n_int = a_mat.shape
    hash_vec = np.random.choice(s_int, n_int, replace=True)
    sign_vec = np.random.choice(2, n_int, replace=True) * 2 - 1
    sketch_a_mat = np.zeros((m_int, s_int))
    for j in range(n_int):
        h = hash_vec[j]
        g = sign_vec[j]
        sketch_a_mat[:, h] += g * a_mat[:, j]
    return sketch_a_mat



@jit('float64[:,:](float64[:,:], int64)',nopython=True)#, parallel=True, fastmath=True)
def _countSketch(data, sketch_dimension):
    '''Sketching as a helper function for the class function call with jit'''
    n,d = data.shape
    sketch = np.zeros((sketch_dimension,d))
    nonzero_rows, nonzero_cols = np.nonzero(data)
    hashedIndices = np.random.choice(sketch_dimension, n, replace=True)
    randSigns = np.random.choice(2, n, replace=True) * 2 - 1
    for ii,jj in zip(nonzero_rows,nonzero_cols):
        bucket = hashedIndices[ii]
        sketch[bucket, jj] += randSigns[ii]*data[ii,jj]
    return sketch

@jit('float64[:,:](int32[:],int32[:],float64[:],int64,int64,int64)',nopython=True)
def _countSketch_fast(nonzero_rows, nonzero_cols, nonzero_data, n, d, sketch_dimension):
    '''Perform count sketch on preprocessed data'''
    sketch = np.zeros((sketch_dimension, d))
    hashedIndices = np.random.choice(sketch_dimension, n, replace=True)
    randSigns = np.random.choice(2, n, replace=True) * 2 - 1
    nnz_id = 0
    old_row = 0
    bucket, sign = hashedIndices[0], randSigns[0]

    for row_id in nonzero_rows:
        col_id = nonzero_cols[nnz_id]
        data_val = nonzero_data[nnz_id]
        if row_id == old_row:
            sketch[bucket, col_id] += sign*data_val
        else:
            bucket = hashedIndices[row_id]
            sign = randSigns[row_id]
            sketch[bucket, col_id] += sign*data_val
            old_row = row_id
        nnz_id += 1
    return sketch



def generate_data(m, n, sigma=5, density=0.2):
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

#@jit(nopython=True)
def ihs_lasso(x,x0,S_A, ATy,covariance_mat, b, _lambda):
    #print("ATy {} ".format(ATy.shape))
    #ATy = np.ravel(ATy)
    #print("ATy {} ".format(ATy.shape))
    #print("cov shape {}".format(covariance_mat.shape))
    #print("b shape {}".format(b.shape))
    norm_term = 0.5*np.linalg.norm(S_A@(x-x0))**2
    inner_product = (x-x0).T@(ATy - covariance_mat@x0)  + (x-x0).T@np.dot(S_A.T, np.dot(S_A,x0))

    return norm_term - inner_product + _lambda*np.linalg.norm(x,1)


def iterative_hessian_lasso(data, targets, regulariser, sketch_size, max_iters):
    '''
    Original problem is min 0.5*||Ax-b||_2^2 + lambda||x||_1
    IHS asks us to minimise 0.5*||SAx||_2^2 - <A^Tb, x> over
    the constraints.

    '''
    setup_time_start= default_timer()
    A = np.asfortranarray(data)
    print(A.flags['F_CONTIGUOUS'])
    y = np.asfortranarray(targets)
    _lambda = regulariser
    n,d = A.shape
    x0 = np.zeros(shape=(d,), order='F')
    m = int(sketch_size) # sketching dimension

    ATy = A.T@y
    print(ATy.flags['F_CONTIGUOUS'])
    covariance_mat = A.T@A

    #old obj for while loop
    old_norm = 1.0
    norm_diff = 1.0
    n_iter = 0

    opt_time = 0
    sketch_time = 0
    setup_end = default_timer() - setup_time_start

    #while(n_iter < max_iters and norm_diff > 10E-6):

    for n_iter in range(max_iters):
        if norm_diff > 1E-3:
            print("ITERATION {}".format(n_iter))

            start_sketch_time = default_timer()
            S_A = _countSketch(A, sketch_size)
            end_sketch_time = default_timer() - start_sketch_time
            print("SKETCH TIME: {}".format(end_sketch_time))
            sketch_time += end_sketch_time


            start = default_timer()
            sub_prob = optimize.minimize(ihs_lasso,x0,
                                      args=(x0, S_A, ATy, covariance_mat, y, _lambda),
                                      #method='SLSQP',
                                      tol=1E-3,
                                      options={'maxiter':10})
            end_opt_time = default_timer()-start
            opt_time += end_opt_time
            print("OPT TIME: {}".format(end_opt_time))
            #print(sub_prob)

            x_new = np.asfortranarray(sub_prob.x)

            norm_diff = np.linalg.norm(x_new - x0)**2/old_norm

            #print(norm_diff)
            x0 = x_new
            old_norm = np.linalg.norm(x0)**2
            n_iter += 1
        else:
            break
    print("Setup cost: {}".format(setup_time))
    return x0, sketch_time, opt_time

def lasso_example(nrows, ncols):
    X, y, coef = generate_data(nrows, ncols, 0.3)
    scale_factor = X.shape[0]**0.5

    X_new = scale_factor*X
    y_new = scale_factor*y
    start = default_timer()
    clf = Lasso()
    clf.fit(X_new,y_new)
    print("SKLEARN SOLVE TIME: {}".format(default_timer() - start))
    x_opt = clf.coef_



    return x_opt

def scaling_times():
    nrows = [200000, 250000, 300000, 400000]#, 500000]# 600000, 700000, 800000, 900000, 1000000] #, 25000, 50000]
    ncols = [100, 150, 200, 250, 300, 400, 500] #, 500] #, 400] #, 500] #, 1500, 2000]

    sklearn_times = {}
    ihs_times = {}

    for n,d in itertools.product(nrows,ncols):
        if d >= n:
            continue
        print('Testing design matrix ({},{})'.format(n, d))

        X,y,coef = generate_data(n, d, 0.3)
        X_sparse = coo_matrix(X)
        X_rows = X_sparse.row
        X_cols = X_sparse.col
        X_data = X_sparse.data

        # SKLEARN
        scale_factor = n**0.5
        X_new = scale_factor*X
        y_new = scale_factor*y
        clf = Lasso(alpha=1.0)
        start = default_timer()
        x_opt = clf.fit(X_new,y_new).coef_
        sklearn_time = default_timer() - start
        sklearn_times[n,d] = sklearn_time

        # IHS method
        ihs_start = default_timer()
        #x_ihs= ihs_lasso_qp(X, X_rows, X_cols, X_data, y, 0.01, 5*d, 20)[0]#, 1.0, 5*d, 20)
        x_ihs= ihs_constrained_qp(X, X_rows, X_cols, X_data, y, 1.0, 3*d, 2)[0]#, 1.0, 5*d, 20)
        ihs_time = default_timer() - ihs_start
        ihs_times[n,d] = ihs_time
        print("SKLEARN TIME: {}".format(sklearn_time))
        print("ERROR: {}".format(\
                            np.linalg.norm(X@(x_opt- x_ihs))**2/np.linalg.norm(X@x_opt)**2))


    #print(sklearn_times)
    fig, ((ax0skl, ax1skl), (ax0ihs, ax1ihs)) = plt.subplots(2,2, figsize=(12,8))

    # need as many as there are columns tested
    skl_times_plot = np.zeros((len(nrows),len(ncols)))
    ihs_times_plot = np.zeros_like(skl_times_plot)

    #print(sklearn_times.keys())
    #print(sklearn_times.items())
    for parameters in sklearn_times.keys():
        #print(parameters)
        #print(sklearn_times[parameters])
        n = int(parameters[0])
        d = int(parameters[1])
        skl_times_plot[nrows.index(n), ncols.index(d)] = sklearn_times[parameters]
        ihs_times_plot[nrows.index(n), ncols.index(d)] = ihs_times[parameters]

    for col in range(skl_times_plot.shape[1]):
        ax0skl.plot(nrows, skl_times_plot[:,col], label=("d = {}".format(ncols[col])))
    ax0skl.legend()
    ax0skl.set_xlabel("Rows")
    ax0skl.set_ylabel("Time (s)")
    ax0skl.set_title("Sklearn")
    ax0skl.grid()


    for row in range(skl_times_plot.shape[0]):
        ax1skl.plot(ncols, skl_times_plot[row, :], label=("n = {}".format(nrows[row])))
    ax1skl.legend()
    ax1skl.set_xlabel("Columns")
    ax1skl.set_ylabel("Time (s)")
    ax1skl.set_title("Sklearn")
    ax1skl.grid()

    for col in range(ihs_times_plot.shape[1]):
        ax0ihs.plot(nrows, ihs_times_plot[:,col], label=("d = {}".format(ncols[col])))
    ax0ihs.legend()
    #ax0ihs.set_yscale('log')
    ax0ihs.set_xlabel("Rows")
    ax0ihs.set_ylabel("Time (s)")
    ax0ihs.set_title("IHS")
    ax0ihs.grid()

    for row in range(ihs_times_plot.shape[0]):
        ax1ihs.plot(ncols, ihs_times_plot[row, :], label=("n = {}".format(nrows[row])))
    ax1ihs.legend()
    ax1ihs.set_xlabel("Columns")
    ax1ihs.set_ylabel("Time (s)")
    ax1ihs.set_title("IHS")
    ax1ihs.grid()
    plt.show()

def test_sketch_time_density():
    ncols = 500
    nrows = 100000
    densities = np.linspace(0.1,1,num=10)
    sketch_size = 1250
    trials = 10
    X, y, coef = generate_data(nrows, ncols, 0.1)
    x = np.random.randn(ncols)
    true_norm = np.linalg.norm(X@x)**2
    X_sparse = coo_matrix(X)
    X_rows = X_sparse.row
    X_cols = X_sparse.col
    X_data = X_sparse.data
    print("original attempt")
    my_sketch_times = np.zeros(trials)
    my_error = 0
    my_sketch_time_start = default_timer()
    for i in range(trials):
        my_sketch_time_start = default_timer()
        sketch = _countSketch(X,sketch_size)
        my_sketch_times[i] = default_timer() - my_sketch_time_start
        my_error += np.abs(np.linalg.norm(sketch@x)**2 - true_norm)/true_norm

    my_sketch_time = np.mean(my_sketch_times)
    #
    XX = X.T
    print('PYRLA method (dense)')
    pyrla_sketch_times = np.zeros(trials)
    pyrla_error = 0
    for i in range(trials):
        pyrla_sketch_time_start = default_timer()
        sketch = countsketch(XX,sketch_size)
        pyrla_sketch_times[i]= default_timer() - pyrla_sketch_time_start
        pyrla_error += np.abs(np.abs(np.linalg.norm(x.T@sketch)**2 - true_norm)/true_norm)
    pyrla_sketch_time = np.mean(pyrla_sketch_times)
    #
    # # sparse sketcher
    print("Sparse sketcher")
    sparse_sketch_times = np.zeros(trials)
    sparse_error = 0

    for i in range(trials):
        sparse_sketch_time_start = default_timer()
        sketch = _countSketch_fast(X_rows, X_cols, X_data, nrows, ncols, sketch_size)
        sparse_sketch_times[i] = default_timer() - sparse_sketch_time_start
        sparse_error += np.abs(np.linalg.norm(sketch@x)**2 - true_norm)/true_norm
    sparse_sketch_time = np.mean(sparse_sketch_times)


    print("my time: {}".format(my_sketch_time))

    print("pyrla time: {}".format(pyrla_sketch_time))
    print("Sparse time: {}".format(sparse_sketch_time))
    print("My average distortion: {}".format(my_error/trials))
    print("Pyrla ave distortion: {}".format(pyrla_error/trials))
    print("Sparse average distortion: {}".format(sparse_error/trials))


    my_time_on_density = []
    pyrla_time_density = []
    sparse_time_density = []

    for density in densities:
        X, y, coef = generate_data(nrows, ncols, density)
        print("DENSITY: {}".format(density))

        print("original attempt")
        my_sketch_times = np.zeros(trials)
        my_error = 0
        my_sketch_time_start = default_timer()
        for i in range(trials):
            my_sketch_time_start = default_timer()
            sketch = _countSketch(X,sketch_size)
            my_sketch_times[i] = default_timer() - my_sketch_time_start
            my_error += np.abs(np.linalg.norm(sketch@x)**2 - true_norm)/true_norm

        my_sketch_time = np.mean(my_sketch_times)
        my_time_on_density.append(my_sketch_time)

        XX = X.T
        print('PYRLA method (dense)')
        pyrla_sketch_times = np.zeros(trials)
        pyrla_error = 0
        for i in range(trials):
            pyrla_sketch_time_start = default_timer()
            sketch = countsketch(XX,sketch_size)
            pyrla_sketch_times[i]= default_timer() - pyrla_sketch_time_start
            pyrla_error += np.abs(np.abs(np.linalg.norm(x.T@sketch)**2 - true_norm)/true_norm)
        pyrla_sketch_time = np.mean(pyrla_sketch_times)
        pyrla_time_density.append(pyrla_sketch_time)

        print("Sparse sketcher")
        sparse_sketch_times = np.zeros(trials)
        sparse_error = 0

        for i in range(trials):
            sparse_sketch_time_start = default_timer()
            sketch = _countSketch_lol(X_rows, X_cols, X_data, nrows, ncols, sketch_size)
            sparse_sketch_times[i] = default_timer() - sparse_sketch_time_start
            sparse_error += np.abs(np.linalg.norm(sketch@x)**2 - true_norm)/true_norm
        sparse_sketch_time = np.mean(sparse_sketch_times)
        sparse_time_density.append(sparse_sketch_time)


    fig, ax = plt.subplots()
    ax.plot(densities, my_time_on_density, label='Original')
    ax.plot(densities, pyrla_time_density, label='PYRLA')
    ax.plot(densities, sparse_time_density, label='Second attempt')
    ax.set_yscale('log')
    ax.set_xlabel("Density (frac nnz)")
    ax.set_ylabel("Summary time")
    ax.legend()
    plt.show()


def main():
    # nrows = 100000
    ncols = 5
    nrows = 10000
    densities = np.linspace(0.1,1,num=10)
    sketch_size = 1250
    trials = 1
    # print("PROFILING LASSO")
    # lp = LineProfiler()
    # lp_wrapper = lp(lasso_example)
    # lp_wrapper(50000, 100)
    # lp.print_stats()
    # print("-"*80)
    # print("PROFILING IHS")
    X, y, coef = generate_data(nrows, ncols, 0.1)
    x = np.random.randn(ncols)
    true_norm = np.linalg.norm(X@x)**2
    X_sparse = coo_matrix(X)
    X_rows = X_sparse.row
    X_cols = X_sparse.col
    X_data = X_sparse.data

    print("Now sketching")

    result = consQP_lasso(X, y, 1.0)
    x = result[2]
    new_x = x[ncols:] - x[:ncols]
    print("new x", -1.0*new_x)
    clf = Lasso(1.0)
    x_opt = clf.fit((X.shape[0])*X,(X.shape[0])*y).coef_
    print("x opt ",x_opt)

    # solver with ihs constrained qp
    x_qp = ihs_constrained_qp(X, X_rows, X_cols, X_data, y, 1.0, sketch_size, 20)
    print("x_qp ", x_qp)

    # solve with the penalty term
    x_ihs = ihs_lasso_qp(X, X_rows, X_cols, X_data, y, 1.0, sketch_size, 20)
    print("x_ihs", x_ihs)

    print(x_opt)
    print(new_x)
    print("SKL/QP approx equal? {}".format(np.testing.assert_array_almost_equal(x_opt, new_x)))
    # nb if the stdout here is 'None' and the file runs then this is ok
    print("new qp error: {}".format(np.linalg.norm(x_opt - x_qp[0])**2))
    print("IHS error: {}".format(np.linalg.norm(x_opt - x_ihs[0])**2))








    # print("-"*80)
    # print("Profiling IHS")
    # lp_ihs = LineProfiler()
    # lp_ihs_wrapper = lp_ihs(ihs_lasso_qp)
    # lp_ihs_wrapper(X, y, 1.0, 1000, 5)
    # lp_ihs.print_stats()
    # print("-"*80)
    # # #
    # # #
    # x_opt = lasso_example(nrows, ncols)
    # qp_start = default_timer()
    # #x_qp = qp_lasso(X,y, 1.0)
    # x_qp = ihs_lasso_qp(X, y, 1.0, 1000, 20)
    # print("QP Solve time: {}".format(default_timer() - qp_start))
    # #print("SKL weights: {}".format(x_opt))
    # #print("IHSQP weights: {}".format(x_qp[0]))
    # print("ERROR: {}".format(np.linalg.norm(X@(x_qp[0] - x_opt))/np.linalg.norm(X@x_opt)))
    # print("Lasso opj: {}".format(np.linalg.norm(X@x_opt - y)**2 + 1.0*np.linalg.norm(x_opt,1)))
    # print("QP obj: {}".format(np.linalg.norm(X@x_qp[0] - y)**2 + 1.0*np.linalg.norm(x_qp[0],1)))
    # start = default_timer()
    # x_time,sk_time, opt_time = iterative_hessian_lasso(X, y, 1.0, 1000, 1)
    # print("Total in parts: {}".format(sk_time+opt_time))
    # print("TOTAL SOLVE TIME: {}".format(default_timer() - start))
    # print("ERROR: {}".format(np.linalg.norm(X@(x_time - x_opt))/np.linalg.norm(X@x_opt)))
    # print("COMPARING TRUE TO ESTIMATED WEIGHTS:")
    # #print(np.c_[x_time[:,None], x_opt[:,None]])
    #
    # print("SCIPY TIMING")
    # scipy_start = default_timer()
    # scipy_result = scipy_lasso(X,y,1.0)
    # print("Scipy time: {}".format(default_timer() - start))
    # scipy_x = scipy_result.x
    #
    # print("SCIPY/SKLEARN ERROR: {}".format(np.linalg.norm(scipy_x - x_opt)))
    # print("SCIPY/IHS ERROR: {}".format(np.linalg.norm(scipy_x - x_time)))
    # print("SKLEARN/IHS ERROR: {}".format(np.linalg.norm(x_opt - x_time)))
    # # print("-"*80)
    # # print("-"*80)
    print("Checking the scaling")
    # scaling_times()




if __name__ =="__main__":
    main()
