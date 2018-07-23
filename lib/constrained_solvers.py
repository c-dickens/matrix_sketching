import numpy as np
import quadprog
import cvxopt as cvx
from sklearn.datasets import make_regression
from  scipy import optimize
import cvxpy

def loss(x, A, b):
    return 0.5*np.linalg.norm(A@x - b)**2

def lasso_solver_qp(hessian, q, ell_1_bound):
    '''helper function to wrap the lasso solver with constraints in matrix
    form.

    Inputs:
        - hessian: d x d numpy array
        - q: inner_product term
        - ell_1_bound: float to bound the solution ||x||_1 <= ell_1_bound

    Output:
        - z: arg-minimiser of the LASSO problem

    Problem form:
    min 0.5*||Ax - b||_2^2 s.t ||x||_1 <= t

    Larger Hessian:
    Q = ( H   - H )
        (- H    H )

    Larger inner product term:
    c = ( q)
        (-q)

    Constraints:
    (I_d  I_d)  <=  (s_d)
    (  I_2d  )  <=  (0_2d)

    QP solver quadprog requires 0.5*x.T*Q*x - c.T*x
    subject to: C.Tx >= b

    Setup taken from
    https://stats.stackexchange.com/questions/119795/quadratic-programming-and-lasso
    '''
    # print("Entering LASSO solver")
    # print("inner prod shape {}".format(q.shape))
    d = hessian.shape[0]
    # Larger Hessian matrix
    Q = np.vstack((np.hstack((hessian, -hessian)),np.hstack((-hessian, hessian)))) + 1E-10*np.identity(2*d)
    print("New Hessian shape {}".format(Q.shape))
    # Larger inner product
    c = np.hstack((q, -1.0*q))
    print("Linear term shape: {}".format(c.shape))
    # Constraints
    constraints = np.hstack((np.identity(d), np.identity(d) ))
    constraints = np.vstack((constraints, -1.0*np.identity(2*d)))

    # Bounds
    b = np.zeros((3*d))

    # mutliply bny -1.0 to fix the less than from setup
    # link to the greater than for the implementation.
    b[:d] = -1.0*ell_1_bound*d
    constraints *= -1.0
    result = quadprog.solve_qp(Q.astype(np.double), c.astype(np.double) , constraints.T, b)
    return result

def lasso_solver_cvx(hessian, q, ell_1_bound):
    '''helper function to wrap the lasso solver with constraints in matrix
    form.

    Inputs:
        - hessian: d x d numpy array
        - q: inner_product term
        - ell_1_bound: float to bound the solution ||x||_1 <= ell_1_bound

    Output:
        - z: arg-minimiser of the LASSO problem

    Problem form:
    min 0.5*||Ax - b||_2^2 s.t ||x||_1 <= t

    Larger Hessian:
    Q = ( H   - H )
        (- H    H )

    Larger inner product term:
    c = ( q)
        (-q)

    Constraints:
    (I_d  I_d)  <=  (s_d)
    (  I_2d  )  <=  (0_2d)

    QP solver quadprog requires 0.5*x.T*Q*x - c.T*x
    subject to: C.Tx >= b

    Setup taken from
    https://stats.stackexchange.com/questions/119795/quadratic-programming-and-lasso
    '''
    print("Entered solver")
    #print(hessian)
    #print(q)
    q *= -1.0  # fixes the negative in objective vs positive in CVX issue
    d = hessian.shape[0]
    # Larger Hessian matrix
    Q = np.vstack((np.hstack((hessian, -hessian)),np.hstack((-hessian, hessian)))) + 1E-10*np.identity(2*d)
    #print("New hessian shape: {}".format(Q.shape))

    # Larger inner product
    c = np.hstack((-1.0*q, q))
    c = c[:, None]
    #print("Shape c {}".format(c.shape))
    # Constraints
    constraints = np.hstack((np.identity(d), np.identity(d) ))
    #print("Constraints shape: {}".format(constraints.shape))
    constraints = np.vstack((constraints, -1.0*np.identity(2*d)))

    # Bounds
    b = np.zeros((3*d,1),dtype=np.float32)

    # mutliply bny -1.0 to fix the less than from setup
    # link to the greater than for the implementation.
    b[:d] = ell_1_bound


    #print("Inner prod shape: {}".format(c.shape))
    #print("Constraints shape: {}".format(constraints.shape))
    #print("Bound shape: {}".format(b.shape))

    # constraints.T as the QP solver does internal transpose.
    Q = cvx.matrix(Q)
    #print(Q)
    c = cvx.matrix(c.astype(np.double))
    #print(c)
    constraints = cvx.matrix(constraints)
    #print(constraints)
    b = cvx.matrix(b.astype(np.double))
    #print(b)
    result = cvx.solvers.qp(Q, c , constraints, b)
    np_result = np.array(result['x'])
    print(np_result)
    x = np_result[:d] - np_result[d:]
    x = np.ravel(x)
    return x

def lasso_solver_cvxpy(hessian, q, ell_1_bound):

    d = hessian.shape[0]
    x = cvxpy.Variable((d,1))
    lasso_bound = cvxpy.Parameter(non)





if __name__ == "__main__":
    np.random.seed(10)
    lasso_bound = [-1.0, 0.1, 0.5, 1.0, 5.0, 10.0, 25.0,  50.0, 75, 100.0, 10000.0]
    syn_data,syn_targets,coef = make_regression(n_samples = 5000, n_features = 2, coef=True)
    syn_data = syn_data.astype(np.float32)
    syn_targets = syn_targets.astype(np.float32)
    print("Data coefficients: {}".format(coef))

    Hessian = syn_data.T@syn_data
    ATy = syn_data.T@syn_targets
    print("Hessian done.")

    # quadprog method
    # for lasso_val in lasso_bound:
    #     result = lasso_solver_qp(Hessian, ATy.T, lasso_val)
    #     print("ell_1_bound {}\n results {}".format(lasso_val, result))


    # CVXPY
    # for lasso_val in lasso_bound:
    #     cvx_result = lasso_solver_cvx(Hessian, ATy.T, lasso_val)
    #     print("Lasso bound: {}, QP estimate: {}".format(lasso_val, cvx_result))
    #
    # # cvxopt
    # for lasso_val in lasso_bound:
    #     cvx_result = lasso_solver_cvx(Hessian, ATy.T, lasso_val)
    #     print("Lasso bound: {}, QP estimate: {}".format(lasso_val, cvx_result))

    # # scipy
    for lasso_val in lasso_bound:
        cons = ({'type' : 'ineq', 'fun' : lambda x :  + np.sum(np.abs(x))})
        #cons = ({'type' : 'eq', 'fun' : lambda x : - 1.0 })
        scipy_result = optimize.minimize(loss,np.zeros(syn_data.shape[1],dtype=np.float32),
                                         args=(syn_data,syn_targets),
                                         constraints=cons)
        print(scipy_result)
        optimal_weights = scipy_result['x']
        print("Lasso bound: {}, Exact soln: {}".format(lasso_val, optimal_weights))
