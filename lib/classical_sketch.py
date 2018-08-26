'''
Class implementation of the classical sketch/sketch and solve method
'''
import numpy as np
import quadprog as qp
from scipy.optimize import minimize
from . import Sketch, CountSketch, SRHT, GaussianSketch
from timeit import default_timer

############HELPER FUNCTIONS for CONSTRAINED REGRESSION #######################
def qp_lasso(data, targets, regulariser):
    d = data.shape[1]
    Q = data.T@data
    big_hessian = np.vstack((np.c_[Q, -Q], np.c_[-Q,Q])) + 1E-10*np.eye(2*d)
    big_linear_term = np.hstack((-targets.T@data, targets.T@data))

    I_d = np.eye(d)
    constraint_matrix = np.vstack((np.eye(2*d), np.c_[I_d, I_d]))
    constraint_vals = np.zeros((3*d))
    constraint_vals[:d] = regulariser
    result = qp.solve_qp(big_hessian, big_linear_term, -constraint_matrix.T, constraint_vals)

    return result

 ##############################################################################

class ClassicalSketch(CountSketch, SRHT, GaussianSketch):
    '''Performs the sketch and solve aka classical sketch method on a regression
    problem by calling a specific sketch method.

    Inherits from random projection classes CountSketch, SRHT, Gaussian.

    References
    -----------
    [1] - https://arxiv.org/abs/1411.4357
    '''

    def __init__(self, data, targets, sketch_dimension, sketch_type,
                                number_iterations=None, random_state=None):

        self.data = data
        self.targets = targets
        self.sketch_type = sketch_type
        self.data_shape = data.shape
        self.sketch_function = {"CountSketch" : CountSketch,
                                "SRHT"        : SRHT,
                                "Gaussian": GaussianSketch}
        self.number_iterations = number_iterations # redundant parameter added for eas of experiments
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

        elif self.sketch_type == "Gaussian":
            if random_state is not None:
                super(GaussianSketch,self).__init__(data, sketch_dimension,\
                                                            random_state)
            else:
                super(GaussianSketch,self).__init__(data, sketch_dimension)

    def sketch(self):
        '''Generates the summary of X = [data, targets] and then splits the
        problem as S_X = S(X), S_data = S(data), S_targets = S(targets).

        Approximation then computed by solving min ||S_data x - S_targets||.

        Guarantees are from subspace embedding property.
        nb. This only works for single right hand sides but the theory can work
        for mutiple so need to adjust the data/target split index.
        '''
        sketch_function = self.sketch_function[self.sketch_type]
        data_targets = np.c_[self.data, self.targets]
        summary = sketch_function(data_targets, self.sketch_dimension)
        #data_target_sketch = summary.sketch(data_targets)
        data_target_sketch = summary.sketch(data_targets)
        sketched_data = data_target_sketch[:,:-1]
        sketched_targets = data_target_sketch[:,-1]
        return sketched_data, sketched_targets

    def solve(self, constraints=None):
        '''Performs the sketch and solve optimization.

        TBC:
        constraints is a dictionary of constraints to pass to the scipy solver
        constraint dict must be of the form:
            constraints = {'type' : 'lasso',
                           'bound' : t}
        where t is an np.float
        '''

        # Step 1: sketch the data and the targets:
        sketched_data, sketched_targets = self.sketch()
        d = self.data_shape[1]
        # Step 2 solve the regression
        if constraints is None:
            sketch_x_hat = np.linalg.lstsq(sketched_data, sketched_targets, rcond=None)[0]
            return sketch_x_hat

        elif constraints['problem'] == 'lasso':
            lasso_bound = constraints['bound']
            sketch_result = qp_lasso(sketched_data, sketched_targets, lasso_bound)
            x_out = sketch_result[0]
            x_hat = x_out[d:] - x_out[:d]

            return x_hat
