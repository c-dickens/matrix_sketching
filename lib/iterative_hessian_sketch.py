'''
Class implementation of the iterative hessian sketch
'''
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import coo_matrix
from . import Sketch, CountSketch, SRHT, GaussianSketch
import quadprog as qp
from timeit import default_timer



############################## SOLVER FUNCTIONS ###############################
def lasso(A,x,b,regulariser):
    return np.linalg.norm(A@x-b)**2 - regulariser*np.linalg.norm(x,1)


def ihs_lasso_solver(sketch, ATy, data, regulariser, old_x):
    '''Solve the iterative version of lasso.  QP constrants adapted from
    https://stats.stackexchange.com/questions/119795/quadratic-programming-and-lasso
    '''
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
    return result




################################################################################



class IHS(CountSketch, SRHT, GaussianSketch):
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
        self.n, self.d = data.shape
        self.sketch_type = sketch_type
        self.number_iterations = number_iterations
        self.data_shape = data.shape
        self.sketch_function = {"CountSketch" : CountSketch,
                                "SRHT"        : SRHT,
                                "Gaussian"    : GaussianSketch}
        X_coo = coo_matrix(self.data)
        self.nonzero_rows = X_coo.row
        self.nonzero_cols = X_coo.col
        self.nonzero_data = X_coo.data
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

    def generate_summaries(self, timing=False):
        '''Generates T = number_iterations sketches of type defined by
        self.sketch_type and returns a 3d numpy array with each sketch being a
        layer in the tensor indexed by its iteration nuber to be called.


        FUTURE WORK
        -----------
        1. CAN THIS BE DONE IN PARALLEL TO SAVE REPEATEDLY LOOPING OVER THE DATA?
        2. CAN WE REPLICATE THIS WITH A TIMING PARAMETER TO SEE HOW LONG THE
        AVERAGE SUMAMRY TIME IS?
        '''


        sketch_function = self.sketch_function[self.sketch_type]

        # Generate a sketch_dim x num_cols in data x T tensor
        all_sketches = np.zeros(shape=(self.sketch_dimension,
                                       self.data_shape[1],
                                       self.number_iterations))
        summary_times = np.zeros(shape=(self.number_iterations,))
        if self.random_state is not None:
            for iter_num in range(self.number_iterations):
                #print("Testing sketch function {}".format(self.sketch_type))

                summary = sketch_function(self.data, self.sketch_dimension, timing=True)
                #end = default_timer() - start

                # if self.sketch_type == "CountSketch":
                #     all_sketches[:,:,iter_num] = summary.sketch()
                # else:
                all_sketches[:,:,iter_num], time = summary.sketch(self.data)
                summary_times[iter_num] = time

            if timing is True:
                return all_sketches, np.mean(summary_times)
            else:
                return all_sketches
        else:
            for iter_num in range(self.number_iterations):
                #print("")
                #print("Iteration {}".format(iter_num))
                summary,time = sketch_function(self.data, self.sketch_dimension,self.random_state, timing=True)
                summary_times[iter_num] = time
                # if self.sketch_type == "CountSketch":
                #     all_sketches[:,:,iter_num] = summary.sketch()
                # else:
                all_sketches[:,:,iter_num] = summary.sketch(self.data)
            if timing is True:
                return all_sketches, np.mean(summary_times)
            else:
                return all_sketches

    def solve(self, constraints=None, timing=False):
        '''constraints is a dictionary of constraints to pass to the scipy solver
        constraint dict must be of the form:
            constraints = {'problem' : 'lasso',
                           'bound' : t}
        where t is an np.float

        timing - bool - False don't return timings, if true then do
        '''

        # Setup
        ATy = np.ravel(self.data.T@self.targets)
        #covariance_matrix = self.data.T@self.data
        summaries = self.generate_summaries()
        x0 = np.zeros(shape=(self.data.shape[1]))
        norm_diff = 1.0
        old_norm = 1.0

        if constraints is None:
            for iter_num in range(self.number_iterations):
                #if norm_diff > 1E-5:
                #print("Entering if part")
                sketch = summaries[:,:, iter_num]
                approx_hessian = sketch.T@sketch
                inner_product = self.data.T@(self.targets - self.data@x0)

                #QP_sol = qp.solve_qp(approx_hessian,inner_product)
                #print("Iteration: {}".format(iter_num))
                #x_new = QP_sol[0]

                z = ATy  - self.data.T@(self.data@x0) + approx_hessian@x0
                sol = np.linalg.lstsq(approx_hessian,z)
                x_new = sol[0]
                x0 = x_new


                # Fractional decrease per iteration check
                #norm_diff = np.linalg.norm(self.data@(x_new - x0))**2/old_norm
                #print(norm_diff)
                #x0 += x_new
                #old_norm = np.linalg.norm(self.data@(x0))**2
            # else:
            #     #print("Break")
            #     break
            return x0

        elif constraints['problem'] == 'lasso':
            lasso_bound = constraints['bound']
            print("Lasso bound: {}".format(lasso_bound))

            setup_time_start = default_timer()
            A = self.data
            y = self.targets
            x0 = np.zeros(shape=(self.d,))
            m = int(self.sketch_dimension)
            ATy = A.T@y
            setup_time = default_timer() - setup_time_start


            old_norm = 1.0
            norm_diff = 1.0
            approx_norm_sum = 0
            old_obj_val = lasso(A,x0,y,lasso_bound)
            # measurable timing vars
            opt_time = 0
            # sketch_time = 0 unnecessary as will come from the previous sketch call



            for n_iter in range(self.number_iterations):
                if norm_diff > 10E-24:
                    print("ITERATION {}".format(n_iter))

                    # Potentially remove this for a single call to sketch from the sketch_function
                    # start_sketch_time = default_timer()
                    S_A = summaries[:,:, n_iter]
                    # end_sketch_time = default_timer() - start_sketch_time
                    # print("SKETCH TIME: {}".format(end_sketch_time))
                    # sketch_time += end_sketch_time


                    ### optimization ###
                    opt_start = default_timer()
                    #sub_prob = lasso_qp_iters(S_A, ATy, covariance_mat, _lambda, x0)
                    sub_prob = ihs_lasso_solver(S_A, ATy, A, lasso_bound, x0)
                    end_opt_time = default_timer()-opt_start
                    opt_time += end_opt_time

                    ### Norm checking and updates ###
                    #x_out = sub_prob[2]
                    x_out = sub_prob[0]

                    x_new = x_out[self.d:] - x_out[:self.d]
                    #print("iterative norm ",np.linalg.norm(x_new))
                    new_obj_val = lasso(A,x_new,y,lasso_bound)
                    norm_diff = np.abs(new_obj_val - old_obj_val)/old_obj_val
                    #print("soln norm ", np.linalg.norm(x_new,1))
                    #approx_norm_sum += np.linalg.norm(x_new,1)
                    print("Norm diff {}".format(norm_diff))
                    x0 += x_new
                    old_obj_val = new_obj_val

            print("Sum of norms: {}".format(approx_norm_sum))
            print("x approx norm ", np.linalg.norm(x0,1))
            print("Setup cost: {}".format(setup_time))
            #print("Total sketching time: {}".format(sketch_time))
            print("Total optimization time: {}".format(opt_time))

            if timing:
                return x0, opt_time
            else:
                return x0
