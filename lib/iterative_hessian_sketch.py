'''
Class implementation of the iterative hessian sketch
'''
import numpy as np
from scipy.optimize import minimize
from . import Sketch, CountSketch, SRHT
#import quadprog
#import cvxopt as cvx
from timeit import default_timer


############################## HELPER FUNCTIONS ###############################
# def lasso_solver(hessian, q, ell_1_bound):
#     '''helper function to wrap the lasso solver with constraints in matrix
#     form.
#
#     Inputs:
#         - hessian: d x d numpy array
#         - q: inner_product term
#         - ell_1_bound: float to bound the solution ||x||_1 <= ell_1_bound
#
#     Output:
#         - z: arg-minimiser of the LASSO problem
#
#     Problem form:
#     min 0.5*||Ax - b||_2^2 s.t ||x||_1 <= t
#
#     Larger Hessian:
#     Q = ( H   - H )
#         (- H    H )
#
#     Larger inner product term:
#     c = ( q)
#         (-q)
#
#     Constraints:
#     (I_d  I_d)  <=  (s_d)
#     (  I_2d  )  <=  (0_2d)
#
#     QP solver quadprog requires 0.5*x.T*Q*x - c.T*x
#     subject to: C.Tx >= b
#
#     Setup taken from
#     https://stats.stackexchange.com/questions/119795/quadratic-programming-and-lasso
#     '''
#     # print("Entering LASSO solver")
#     # print("inner prod shape {}".format(q.shape))
#     d = hessian.shape[0]
#     # Larger Hessian matrix
#     Q = np.vstack((np.hstack((hessian, -hessian)),np.hstack((-hessian, hessian)))) + 1E-10*np.identity(2*d)
#
#     # Larger inner product
#     c = np.hstack((q, -1.0*q))
#     # Constraints
#     constraints = np.hstack((np.identity(d), np.identity(d) ))
#     constraints = np.vstack((constraints, -1.0*np.identity(2*d)))
#
#     # Bounds
#     b = np.zeros((3*d))
#
#     # mutliply bny -1.0 to fix the less than from setup
#     # link to the greater than for the implementation.
#     b[:d] = -1.0*ell_1_bound
#     constraints *= -1.0
#
#     # print("New hessian shape: {}".format(Q.shape))
#     # print("Inner prod shape: {}".format(c.shape))
#     # print("Cosntraints shape: {}".format(constraints.shape))
#     # print("Bound shape: {}".format(b.shape))
#
#     # constraints.T as the QP solver does internal transpose.
#     result = quadprog.solve_qp(Q, c , constraints.T, b)
#     return result




################################################################################



class IHS(CountSketch, SRHT):
    '''Performs the iterative hessian sketch method by calling a
    specific random projection.

    Inherits from both CountSketch and SRHT classes.

    References
    ----------
    [1] - https://arxiv.org/abs/1411.0347  Pilanci & Wainwright
    '''


    def __init__(self, data, targets, sketch_dimension, sketch_type, number_iterations,
                                    random_state=None):
        '''
        Parameters:
        -------


        Notes:
        -------
        sketch_type tells the class which sketch to instantiate but we also
        instantiate a dictionary to correctly call the correct sketch class.
        '''
        self.data = data
        self.targets = targets
        self.sketch_type = sketch_type
        self.number_iterations = number_iterations
        self.data_shape = data.shape
        self.sketch_function = {"CountSketch" : CountSketch,
                                "SRHT"        : SRHT}
        if self.sketch_type == "CountSketch":
            if random_state is not None:
                super(CountSketch,self).__init__(data, sketch_dimension,\
                                                            random_state)
            else:
                super(CountSketch,self).__init__(data, sketch_dimension)

        elif self.sketch_type == "SRHT":
            if random_state is not None:
                super(SRHT,self).__init__(data, sketch_dimension,\
                                                            random_state)
            else:
                super(SRHT,self).__init__(data, sketch_dimension)

    def generate_summaries(self):
        '''Generates T = number_iterations sketches of type defined by
        self.sketch_type and returns a 3d numpy array with each sketch being a
        layer in the tensor indexed by its iteration nuber to be called.


        FUTURE WORK
        -----------
        1. CAN THIS BE DONE IN PARALLEL TO SAVE REPEATEDLY LOOPING OVER THE DATA?
        2. CAN WE REPLICATE THIS WITH A TIMING ARGUMENT TO SEE HOW LONG THE
        AVERAGE SUMAMRY TIME IS?
        '''


        sketch_function = self.sketch_function[self.sketch_type]

        # Generate a sketch_dim x num_cols in data x T tensor
        all_sketches = np.zeros(shape=(self.sketch_dimension,
                                       self.data_shape[1],
                                       self.number_iterations))

        if self.random_state is not None:
            for iter_num in range(self.number_iterations):
                #print("")
                #print("Iteration {}".format(iter_num))
                summary = sketch_function(self.data, self.sketch_dimension)
                all_sketches[:,:,iter_num] = summary.sketch(self.data)
            return all_sketches
        else:
            for iter_num in range(self.number_iterations):
                #print("")
                #print("Iteration {}".format(iter_num))
                summary = sketch_function(self.data, self.sketch_dimension,self.random_state)
                all_sketches[:,:,iter_num] = summary.sketch(self.data)
            return all_sketches

    # def solve(self, constraints=None):
    #     '''constraints is a dictionary of constraints to pass to the scipy solver
    #     constraint dict must be of the form:
    #         constraints = {'type' : 'lasso',
    #                        'bound' : t}
    #     where t is an np.float
    #     '''
    #
    #     # Setup
    #     ATy = np.ravel(self.data.T@self.targets)
    #     #covariance_matrix = self.data.T@self.data
    #     summaries = self.generate_summaries()
    #     x0 = np.zeros(shape=(self.data.shape[1]))
    #
    #
    #     if constraints is None:
    #         for iter_num in range(self.number_iterations):
    #             sketch = summaries[:,:, iter_num]
    #             approx_hessian = sketch.T@sketch
    #             inner_product = self.data.T@(self.targets - self.data@x0)
    #
    #             QP_sol = quadprog.solve_qp(approx_hessian,inner_product)
    #             print("Iteration: {}".format(iter_num))
    #             x0 += QP_sol[0]
    #     elif constraints['type'] == 'lasso':
    #         lasso_bound = constraints['bound']
    #         print("Lasso bound: {}".format(lasso_bound))
    #         for iter_num in range(self.number_iterations):
    #             sketch = summaries[:,:, iter_num]
    #             approx_hessian = sketch.T@sketch
    #             inner_product = self.data.T@(self.targets - self.data@x0)
    #             QP_sol = lasso_solver(approx_hessian, inner_product, lasso_bound)
    #             print(QP_sol)
    #
    #             QP_out = QP_sol[0]
    #             #print(QP_out)
    #             #print(QP_out[:self.data.shape[1]])
    #             #print(QP_out[self.data.shape[1]:])
    #             x_out = QP_out[:self.data.shape[1]] - QP_out[self.data.shape[1]:]
    #             #print("x_out: {}".format(x_out))
    #             x0 = x_out
    #     return x0
