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
import cvxpy as cp



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

def cvxpy_lasso_norm(sketch, data, targets, x0, _lambda):
    '''Solve the IHS constrained regression problem which is as follows:
    [*]    0.5*||SA(x-x0)||^2 - (x-x0).T*[A.T*(b-Ax0)] + \lambda*||x-x0||_1

    Inputs:
        sketch - SA in above formulation
        data - A
        targets - b
        x - variable for cvxpy
        x0 - current x value for offset
        lambda - lambda, can be a cp.Parameter or constant
    output:
        x_new - minimises the problem [*]
    '''
    n,d = data.shape
    dataTtargets = data.T@targets
    sub_x = cp.Variable(d)

    norm_term = 0.5*cp.sum_squares(sketch@(sub_x-x0)) # <==> 0.5*||SA(x-x0)||^2
    linear_term = (sub_x-x0).T@(dataTtargets-data.T@(data@x0))
    regulariser = _lambda*cp.norm(sub_x-x0,1)

    # Setup cvxpy problem
    sub_obj = cp.Minimize(norm_term - linear_term + regulariser)
    sub_prob = cp.Problem(sub_obj)
    sub_prob.solve()

    return sub_x.value # Also want to return objective value

def ihs_lasso_cvxpy(data, targets, sketch_dimension, _lambda, max_iters):
    '''Perform ihs iterations using the countsketch'''
    x0 = np.zeros(data.shape[1])
    itr = 0
    while(itr < max_iters):
        print(itr)

        # 1. Generate sketch then solve subproblem
        summary = CountSketch(data, sketch_dimension)
        sketch = summary.sketch(data)

        # 2. Solve the minimisation
        x_new = cvxpy_lasso_norm(sketch, data, targets, x0, _lambda)
        x0 += x_new
        itr += 1
    return x0


# subset_size = 50008
#sketch_size = 1000
# X = rawdata_mat[:subset_size, 1:11]
# y = rawdata_mat[:subset_size, 0]
# y = y[:,None]
print("generating data")
X,y,beta = generate_data(10000,5,5,0.01)
n,d = X.shape
print("Shape of data: {}".format(rawdata_mat.shape))
print("Shape of testing data: {}".format(X.shape))
print("Shape of test vector: {}".format(y.shape))
repeats = 3


def main():

    # solve the lasso problem in sklearn
    lassoModel = Lasso(alpha=2.0,max_iter=1000)
    sklearn_X, sklearn_y = X.shape[0]*X, X.shape[0]*y
    sklearn_time = np.zeros(repeats)
    print("-"*80)
    optimal_weights = np.zeros(d)

    for rep in range(repeats):
        start = default_timer()
        lassoModel.fit(sklearn_X,sklearn_y)
        sklearn_time[rep] = default_timer() - start
        optimal_weights += lassoModel.coef_
    skl_optimal_weights = optimal_weights/repeats
    print("SKLEARN TIME: {}".format(np.mean(sklearn_time, dtype=np.float64)))
    print("SKLEARN weights: {}".format(skl_optimal_weights))
    sklearn_objective = np.linalg.norm(X@skl_optimal_weights - y)**2\
                                        + 2.0*np.linalg.norm(skl_optimal_weights,1)
    print("SKLEARN OBJECTIVE: {}".format(sklearn_objective))
    print("-"*80)

    ################## Solve the problemn using CVXPY ########################
    x_cvxpy = cp.Variable(d)
    gamma = cp.Parameter(nonneg=True)
    error = cp.sum_squares(X*x_cvxpy - y)

    # Measurables
    cvxpy_solve_time = np.zeros(repeats)
    cvxpy_setup_time = np.zeros_like(cvxpy_solve_time)
    #solver_methods = cp.installed_solvers()
    #print(solver_methods)
    # remove solvers which throw errors - machine dependent
    # solver_methods.remove('CVXOPT')
    # solver_methods.remove('GLPK')
    # solver_methods.remove('GLPK_MI')
    # solver_methods.remove('SCS')
    # solver_methods.remove('CPLEX')




    #for solver_method in solver_methods:
    print("-"*80)
    print("-"*80)
    print("Solving in full form")
    #print("USING SOLVER: {}".format(solver_method))
    #solver_method = 'OSQP'
    for rep in range(repeats):
        setup_start = default_timer()
        objective = cp.Minimize(error + 2.0*cp.norm(x_cvxpy, 1))
        cvxpy_prob = cp.Problem(objective)
        cvxpy_setup_time[rep] = default_timer() - setup_start
        cvxpy_start = default_timer()
        # cvxpy_prob.solve(solver=solver_method)
        cvxpy_prob.solve()
        cvxpy_solve_time[rep] = default_timer() - cvxpy_start

    print("-"*80)

    print("CVXPY SETUP TIME: {}".format(np.mean(cvxpy_setup_time, dtype=np.float64)))
    print("CVXPY SOLVE TIME: {}".format(np.mean(cvxpy_solve_time, dtype=np.float64)))
    print("CVXPY OBJECTIVE: {}".format(objective.value))
    print("Weights: {}".format(x_cvxpy.value))
    machine_diff = (1/n)*np.linalg.norm(X@(skl_optimal_weights - x_cvxpy.value))**2
    print("CVXPY-normal\SKLEARN DIFF: {}".format(machine_diff))

    ###########################################################################

    ################## Solve the problem as a QP ########################
    print("-"*80)
    print("-"*80)
    print("Solving as a Quadratic Program")
    print("-"*80)
    hessian = X.T@X
    x_qp = cp.Variable(d)
    qp_error = 0.5*cp.quad_form(x_qp, hessian) - (X@x_qp).T@y + y.T@y

    # Times
    qp_setup_time = np.zeros_like(cvxpy_solve_time)
    qp_solve_time = np.zeros_like(cvxpy_solve_time)

    #for solver_method in solver_methods:
    print("-"*80)
    #print("USING SOLVER: {}".format(solver_method))
    for rep in range(repeats):
        setup_start = default_timer()
        qp_objective = cp.Minimize(qp_error + 2.0*cp.norm(x_cvxpy, 1))
        qp_prob = cp.Problem(qp_objective)
        qp_setup_time[rep] = default_timer() - setup_start
        qp_start = default_timer()
        #qp_prob.solve(solver=solver_method)
        qp_prob.solve()
        qp_solve_time[rep] = default_timer() - cvxpy_start

    print("QP SETUP TIME: {}".format(np.mean(qp_setup_time, dtype=np.float64)))
    print("QP SOLVE TIME: {}".format(np.mean(qp_solve_time, dtype=np.float64)))
    print("QP Error: {}".format(qp_objective.value))
    print("Weights: {}".format(x_qp.value))
    machine_diff = (1/n)*np.linalg.norm(X@(skl_optimal_weights - x_qp.value))**2
    print("CVXPYQP\SKLEARN DIFF: {}".format(machine_diff))


    print("-"*80)
    print("-"*80)
    print("Iterative Hessian Sketch")
    print("-"*80)
    print("-"*80)

    x_ihs = ihs_lasso_cvxpy(data=X,targets=y,sketch_dimension=1000,\
                                                    _lambda=2.0, max_iters=20)
    ihs_diff = (1/n)*np.linalg.norm(X@(skl_optimal_weights - x_ihs))**2
    print("IHS\SKLEARN DIFF: {}".format(ihs_diff))
    print(x_ihs)





if __name__ == "__main__":
    main()
    '''
    Comments:
    if solving using the minimize sum of squares then OSQP is fastest.
    However, if solving as a QP then ECOS is fastest - need to be in venv for
    this to work though.
    Also need to play around with which solvers to remove.

    Solving as sum of squares is faster than as a qp but don't know yet how this
    works for IHS.
    '''
