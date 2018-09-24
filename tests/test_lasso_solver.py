'''Tests the countsketch.py methods'''
from timeit import default_timer
import numpy as np
import scipy as sp
import pandas as pd
import unittest
import sys
sys.path.append("..")
import lib
from lib import Sketch
from lib import CountSketch
from lib import iterative_hessian_sketch as ihs
from scipy.sparse import random
import quadprog as qp
import qpsolvers as qps
import cvxopt as cp

# Methods to test
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
from scipy import optimize
from scipy import sparse
from experiment_parameter_grid import ihs_sketches


#################
#random_seed = 10
#np.random.seed(random_seed)

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

#### LASSO QP Solvers.
def cvxopt_lasso(data,targets, constraint):
    '''solve using cvxopt'''
    n,d = data.shape
    Q = data.T@data
    c = data.T@targets

    # Expand the problem
    big_Q = np.vstack((np.c_[Q, -1.0*Q], np.c_[-1.0*Q, Q]))
    big_c = np.concatenate((c,-c))

    # penalty term
    constraint_term = constraint*np.ones((2*d,))
    big_linear_term = constraint_term - big_c

    # nonnegative constraints
    G = -1.0*np.eye(2*d)
    h = np.zeros((2*d,))

    P = cp.matrix(big_Q)
    q = cp.matrix(big_linear_term)
    G = cp.matrix(G)
    h = cp.matrix(h)


    res = cp.solvers.qp(P,q,G,h)
    w = np.squeeze(np.array(res['x']))
    w[w < 1E-8] = 0
    x = w[:d] - w[d:]
    return(x)

def iterative_lasso(sketch_data, data, targets, x0, penalty):
    '''solve the lasso through repeated calls to a smaller quadratic program'''

    # Deal with constants
    n,d = data.shape
    #Q = data.T@data
    Q = sketch_data.T@sketch_data
    c = Q@x0 + data.T@(targets - data@x0)  # data.T@(targets - data@x0)

    # Expand the problem
    big_Q = np.vstack((np.c_[Q, -1.0*Q], np.c_[-1.0*Q, Q]))
    big_c = np.concatenate((c,-c))

    # penalty term
    constraint_term = penalty*np.ones((2*d,))
    big_linear_term = constraint_term - big_c

    # nonnegative constraints
    G = -1.0*np.eye(2*d)
    h = np.zeros((2*d,))

    P = cp.matrix(big_Q)
    q = cp.matrix(big_linear_term)
    G = cp.matrix(G)
    h = cp.matrix(h)


    res = cp.solvers.qp(P,q,G,h)
    w = np.squeeze(np.array(res['x']))
    #w[w < 1E-8] = 0
    x = w[:d] - w[d:]
    return(x)



def main():
    rawdata_mat = np.load('../data/Complex.npy')

    n = 50000
    d = 15
    X,y,x_star = generate_data(n,d,5)
    repeats = 1

    # solve the lasso problem in sklearn
    lasso_penalty_term = 5.0
    lassoModel = Lasso(alpha=lasso_penalty_term ,max_iter=1000)
    sklearn_X, sklearn_y = np.sqrt(n)*X, np.sqrt(n)*y
    sklearn_time = np.zeros(repeats)
    print("-"*80)
    for rep in range(repeats):
        start = default_timer()
        lassoModel.fit(sklearn_X,sklearn_y)
        sklearn_time[rep] = default_timer() - start
    # print("SKLEARN TIME: {}".format(np.mean(sklearn_time, dtype=np.float64)))
    x_opt = lassoModel.coef_

    #print(x_opt)
    print("-"*80)

    x_cvx = cvxopt_lasso(X,y,lasso_penalty_term)
    print()
    np.testing.assert_array_almost_equal(x_cvx, x_opt,5)
    print("QP formulation agrees with Sklearn √√√")


    x0 = np.zeros((d,))
    for i in range(20):
        S_X = (1/np.sqrt(10*d))*np.random.randn(10*d,n)@X
        new_x = iterative_lasso(S_X,X,y,x0,lasso_penalty_term)
        sol_error = np.log((1/n)*np.linalg.norm(new_x-x_opt)**2)
        print("Error after {} iters to opt: {}".format(i+1,sol_error))
        #print(np.c_[new_x[:,None], x_opt[:,None]])
        x0 = new_x

    np.testing.assert_array_almost_equal(x0, x_opt, 6)
    print("Iterative scheme agrees with the the QP formulation √√√")



if __name__ == "__main__":
    main()
