'''Tests the countsketch.py methods'''
from timeit import default_timer
import numpy as np
import pandas as pd
import unittest
import sys
sys.path.append("..")
import lib
from lib import Sketch
from lib import CountSketch
from scipy.sparse import random

# Methods to test
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from scipy import optimize



#################
random_seed = 10
np.random.seed(random_seed)
dir = '..'

rawdata_mat = np.load(dir + '/data/YearPredictionMSD.npy')

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

# subset_size = 50008
sketch_size = 1000
# X = rawdata_mat[:subset_size, 1:11]
# y = rawdata_mat[:subset_size, 0]
# y = y[:,None]
print("generating data")
X,y,beta = generate_data(10000, 2,5,0.01)
print("Shape of data: {}".format(rawdata_mat.shape))
print("Shape of testing data: {}".format(X.shape))
print("Shape of test vector: {}".format(y.shape))
repeats = 5





def ihs_lasso(x,x0,S_A, ATy,covariance_mat, b, _lambda):
    #print("ATy {} ".format(ATy.shape))
    #ATy = np.ravel(ATy)
    #print("ATy {} ".format(ATy.shape))
    #print("cov shape {}".format(covariance_mat.shape))
    #print("b shape {}".format(b.shape))
    norm_term = 0.5*np.linalg.norm(S_A@(x-x0))**2
    inner_product = (x-x0).T@(ATy - covariance_mat@x0)  + (x-x0).T@np.dot(S_A.T, np.dot(S_A,x0))
    #print("IP shape {}".format(inner_product.shape))
    regulariser = _lambda*np.linalg.norm(x-x0,1)
    #print("Reg shape {}".format(regulariser.shape))
    output = norm_term - inner_product + regulariser
    #print("Out shape {}".format(output.shape))
    return norm_term - inner_product + regulariser

def iterative_hessian_lasso(data, targets, regulariser,sketch_size, num_iters):
    '''
    Original problem is min 0.5*||Ax-b||_2^2 + lambda||x||_1
    IHS asks us to minimise 0.5*||SAx||_2^2 - <A^Tb, x> over
    the constraints.

    Does plugging the constrain into the minimisation term just work?'''

    A = data
    y = targets
    _lambda = regulariser
    n,d = A.shape
    x0 = np.zeros(shape=(d,))
    m = int(sketch_size) #sketching dimension

    print("Generating constants")
    ATy_time = default_timer()
    ATy = np.ravel(A.T@y)
    print("ATy time: {}".format(default_timer() - ATy_time))

    cov_mat_time = default_timer()
    covariance_mat = A.T.dot(A)
    print("Cov mat time: {}".format(default_timer() - cov_mat_time))

    # Measurables
    summary_time = np.zeros(num_iters)
    update_time = np.zeros_like(summary_time)

    for n_iter in range(int(num_iters)):
        #print(n_iter)
        #print("x0 shape {}".format(x0.shape))
        all_sketches = np.zeros(shape=(sketch_size,
                                       A.shape[1],
                                       num_iters))



        summary = CountSketch(data=A, sketch_dimension=sketch_size)

        summary_start = default_timer()
        sketch = summary.sketch(A)
        summary_time[n_iter] = default_timer() - summary_start
        print("Iteration {}, time {}".format(n_iter,summary_time[n_iter]))

        update_start = default_timer()
        x_new = optimize.minimize(ihs_lasso,x0,
                                  args=(x0, sketch, ATy, covariance_mat, y, _lambda),
                                  method='BFGS').x
        update_time[n_iter] = default_timer() - update_start






        #
        #start = default_timer()
        #sketch = summary.sketch(X)

        #S_A = sketch_method(A, sketch_size)
        #start = default_timer()

        #solve_time = default_timer() - start
        #print("SOLVE TIME: {}".format(solver_time))
        #print(x_new)
        x0 = x_new

    print("MEAN UPDATE TIME IN IHS: {}".format(np.mean(update_time)))
    print("MEAN SUMMARY TIME IN IHS: {}".format(np.mean(summary_time)))
    print("TOTAL SUMMARY TIME IN IHS: {}".format(np.sum(summary_time)))
    return np.ravel(x0)



def main():

    # solve the lasso problem in sklearn
    lassoModel = Lasso(alpha=2.0,max_iter=1000)
    sklearn_X, sklearn_y = X.shape[0]*X, X.shape[0]*y
    sklearn_time = np.zeros(repeats)
    print("-"*80)
    for rep in range(repeats):
        start = default_timer()
        lassoModel.fit(sklearn_X,sklearn_y)
        sklearn_time[rep] = default_timer() - start
    print("SKLEARN TIME: {}".format(np.mean(sklearn_time, dtype=np.float64)))
    print("-"*80)


    # Sketch the problem and measure time
    sketch_time = np.zeros_like(sklearn_time)
    # nb calcs ignore the first iter for start up due to jit in numba
    for rep in range(repeats):
        summary = CountSketch(data=X, sketch_dimension=sketch_size)
        start = default_timer()
        sketch = summary.sketch(X)
        sketch_time[rep] = default_timer() - start
        print("Iteration {}, time {}".format(rep,sketch_time[rep]))
    print("MEAN SUMMARY TIME ON ({},{},{}): {}".format(X.shape[0], X.shape[1],\
                                              sketch_size,np.mean(sketch_time[1:])))
    print("Sketch shape: {}".format(sketch.shape))
    print("TOTAL SKETCH TIME: {}".format(np.sum(sketch_time[1:])))



    # Now let's try with scipy solver.
    print("-"*80)

    start = default_timer()
    x_ihs = iterative_hessian_lasso(X,y, 2.0,2000,20)
    ihs_time  = default_timer() - start
    print("IHS TIME: {}".format(ihs_time))


    # Comparison to optimal weights
    print("-"*80)
    true_weights = lassoModel.coef_
    print("NORM OF DIFFERENCE: {}".format(np.linalg.norm(X@(true_weights - x_ihs)*(1/X.shape[0]))))
    print("True coefs: {}".format(lassoModel.coef_))
    print("Approx coefs: {}".format(x_ihs))


if __name__ == "__main__":
    main()
