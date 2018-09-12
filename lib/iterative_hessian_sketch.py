'''
Class implementation of the iterative hessian sketch
'''
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import coo_matrix
from . import Sketch, CountSketch, SRHT, GaussianSketch
from hadamard import fastwht
import quadprog as qp
from timeit import default_timer
from numba import jit



############################## SOLVER FUNCTIONS ###############################

def lasso(A,x,b,regulariser):
    return np.linalg.norm(A@x-b)**2 - regulariser*np.linalg.norm(x,1)


def ihs_lasso_solver(sketch, ATy, data, regulariser, old_x, timing=False):
    '''Solve the iterative version of lasso.  QP constrants adapted from
    https://stats.stackexchange.com/questions/119795/quadratic-programming-and-lasso

    if timing
    return result - the optimization output
           solve_time - optimization time
           norm(linear_term) - norm of the gradient
    '''
    d = data.shape[1]
    Q = data.T@data
    big_hessian = np.vstack((np.c_[Q, -Q], np.c_[-Q,Q])) + 1E-5*np.eye(2*d)
    # print("Shape ATy {}".format(ATy.shape))
    # print("Othjer term shape {}".format((data@old_x).shape))
    # print("old_X shape {}".format(old_x.shape))
    # print("data.shape {}".format(data.shape))
    linear_term = ATy - data.T@(data@old_x)
    big_linear_term = np.hstack((-linear_term, linear_term))

    I_d = np.eye(d)
    constraint_matrix = np.vstack((np.eye(2*d), np.c_[I_d, I_d]))
    constraint_vals = np.zeros((3*d))
    constraint_vals[:d] = regulariser

    solve_start = default_timer()
    result = qp.solve_qp(big_hessian, big_linear_term, -constraint_matrix.T, constraint_vals)
    solve_time = default_timer() - solve_start

    if timing:
        return result, solve_time, np.linalg.norm(linear_term)
    else:
        return result

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

@jit(nopython=True)
def countSketch_dense(data, sketch_dimension):
    '''
    count sketch for dense data.
    Views the matrix row-wise and hashes rows to buckets in S.
    '''

    n,d = data.shape
    sketch = np.zeros((sketch_dimension,d))
    hashedIndices = np.random.choice(sketch_dimension, n, replace=True) # hash n rows into sketch dimension buckets
    randSigns = np.random.choice(2, n, replace=True) * 2 - 1
    data = randSigns.reshape(n,1) * data # flip the sign of half of the rows
    for ii in range(sketch_dimension):
        bucket_id = (hashedIndices == ii) # gets row indices in bucket ii
        #bucket_id = np.where(hashedIndices == ii)
        sketch[ii,:] = np.sum(data[bucket_id,:],axis=0) # sums all of the rows in the same bucket

def shift_bit_length(x):
    '''Given int x find next largest power of 2.
    If x is a power of 2 then x is returned '''
    return 1<<(x-1).bit_length()

def srht_transform1(input_matrix, sketch_size, seed=None):
    '''Generate a sketching matrix S to reduce the sample count (i.e. sketching
    on the LHS of data) via the Subsampled Hadamard Transform.

    Given an input_matrix ``A`` of size ``(n, d)``, compute a matrix ``A'`` of
    size (sketch_size, d) which holds:
    .. math:: ||Ax|| = (1 \pm \epsilon)||A'x||
    with high probability. [1]
    The error is related to the number of rows of the sketch and it is bounded

    tbc


    Parameters
    ----------
    input_matrix: array_like
        Input matrix, of shape ``(n, d)``.
    sketch_size: int
        Number of rows for the sketch.
    seed : None or int or `numpy.random.RandomState` instance, optional
        This parameter defines the ``RandomState`` object to use for drawing
        random variates.
        If None (or ``np.random``), the global ``np.random`` state is used.
        If integer, it is used to seed the local ``RandomState`` instance.
        Default is None.
    Returns
    -------
    S_A : array_like
        Sketch of the input matrix ``A``, of size ``(sketch_size, d)``.

    Notes
    -------
    This implementation of the SRHT is fast up to the line which requires the
    copying of the fastmat PRODUCT type to a NumPy array [2].
    The fastmat library is used to quickly compute D*A and the fht library is
    used to compute the product H(DA) quickly by exploiting the FFT [3].

    References
    -------------
    [1] - https://arxiv.org/abs/1411.4357
    [2] - https://github.com/EMS-TU-Ilmenau/fastmat
    [3] - https://github.com/nbarbey/fht



    '''
    nrows = input_matrix.shape[0]
    diag = np.random.choice([1,-1], nrows)
    diag = diag[:,None]
    # print("diag shape: {}".format(diag.shape))
    # print("input mat shape: {}".format(input_matrix.shape))
    signed_mat = diag*input_matrix
    # print(signed_mat.shape)
    S = fastwht(signed_mat)*shift_bit_length(nrows) # shift bit length is normalising factor
    sample = np.random.choice(nrows, sketch_size, replace=False)
    #sample.sort()
    # number from num_rows_data universe
    S = S[sample]
    S = (sketch_size)**(-0.5)*S
    return S






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
                data_rows=None,data_cols=None,data_vals=None,random_state=None,
                timing=False):
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
        self.sketch_dimension = sketch_dimension
        self.n, self.d = data.shape
        self.sketch_type = sketch_type
        self.number_iterations = number_iterations
        self.data_shape = data.shape
        # self.sketch_function = {"CountSketch" : CountSketch,
        #                         "SRHT"        : SRHT,
        #                         "Gaussian"    : GaussianSketch}
        self.sketch_function = {"CountSketch" : _countSketch_fast,
                                "SRHT"        : srht_transform1,
                                "Gaussian"    : GaussianSketch}
        if np.asarray([data_rows,data_cols,data_vals]).any() == None:
            print("Convert to coo mat within IHS")
            X_coo = coo_matrix(self.data)
            self.nonzero_rows = X_coo.row
            self.nonzero_cols = X_coo.col
            self.nonzero_data = X_coo.data
        else:
            print("Already in coo format")
            self.nonzero_rows = data_rows
            self.nonzero_cols = data_cols
            self.nonzero_data = data_vals

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
        AVERAGE SUMMARY TIME IS?
        '''


        sketch_function = self.sketch_function[self.sketch_type]
        #print("using {}".format(sketch_function))
        # Generate a sketch_dim x num_cols in data x T tensor
        all_sketches = np.zeros(shape=(self.sketch_dimension,
                                       self.data_shape[1],
                                       self.number_iterations))
        summary_times = np.zeros(shape=(self.number_iterations,))
        if self.random_state is not None:
            for iter_num in range(self.number_iterations):
                #print("Testing sketch function {}".format(self.sketch_type))

                if self.sketch_type is "CountSketch":
                    #sketch_start = default_timer()
                    all_sketches[:,:,iter_num] = _countSketch_fast(self.nonzero_rows,self.nonzero_cols,self.nonzero_data, self.n, self.d, self.sketch_dimension)
                    #sketch_time = default_timer() - sketch_start
                    #all_sketches[:,:,iter_num] = summary
                elif self.sketch_type is "SRHT":
                    all_sketches[:,:,iter_num] = sketch_function(self.data, self.sketch_dimension)
                else:
                    summary = sketch_function(self.data, self.sketch_dimension)
                    all_sketches[:,:,iter_num] = summary.sketch(self.data)
                #summary_times[iter_num] = time

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
    #
    #     elif constraints['problem'] == 'lasso':
    #         lasso_bound = constraints['bound']
    #         print("Lasso bound: {}".format(lasso_bound))
    #
    #         setup_time_start = default_timer()
    #         A = self.data
    #         y = self.targets
    #         x0 = np.zeros(shape=(self.d,))
    #         m = int(self.sketch_dimension)
    #         ATy = A.T@y
    #         setup_time = default_timer() - setup_time_start
    #
    #
    #         old_norm = 1.0
    #         norm_diff = 1.0
    #         approx_norm_sum = 0
    #         old_obj_val = lasso(A,x0,y,lasso_bound)
    #         # measurable timing vars
    #         opt_time = 0
    #         # sketch_time = 0 unnecessary as will come from the previous sketch call
    #
    #
    #
    #         for n_iter in range(self.number_iterations):
    #             if norm_diff > 10E-10:
    #                 print("ITERATION {}".format(n_iter))
    #
    #                 # Potentially remove this for a single call to sketch from the sketch_function
    #                 # start_sketch_time = default_timer()
    #                 S_A = summaries[:,:, n_iter]
    #                 # end_sketch_time = default_timer() - start_sketch_time
    #                 # print("SKETCH TIME: {}".format(end_sketch_time))
    #                 # sketch_time += end_sketch_time
    #
    #
    #                 ### optimization ###
    #                 opt_start = default_timer()
    #                 #sub_prob = lasso_qp_iters(S_A, ATy, covariance_mat, _lambda, x0)
    #                 sub_prob = ihs_lasso_solver(S_A, ATy, A, lasso_bound, x0)
    #                 end_opt_time = default_timer()-opt_start
    #                 opt_time += end_opt_time
    #
    #                 ### Norm checking and updates ###
    #                 #x_out = sub_prob[2]
    #                 x_out = sub_prob[0]
    #
    #                 x_new = x_out[self.d:] - x_out[:self.d]
    #
    #                 #print("iterative norm ",np.linalg.norm(x_new))
    #                 new_obj_val = lasso(A,x_new,y,lasso_bound)
    #                 norm_diff = np.abs(new_obj_val - old_obj_val)/old_obj_val
    #                 #print("soln norm ", np.linalg.norm(x_new,1))
    #                 #approx_norm_sum += np.linalg.norm(x_new,1)
    #                 #print("Norm diff {}".format(norm_diff))
    #                 x_test_approx = x0.copy() # test diference between old x0 --. now x_test_approx and new x0
    #                 x0 += x_new
    #                 #print("new_x_approx: {}".format(x0))
    #                 print("Norm diff: {}".format(np.linalg.norm(x0 - x_test_approx)))
    #                 old_obj_val = new_obj_val
    #
    #         #print("Sum of norms: {}".format(approx_norm_sum))
    #         print("x approx norm ", np.linalg.norm(x0,1))
    #         print("Setup cost: {}".format(setup_time))
    #         #print("Total sketching time: {}".format(sketch_time))
    #         print("Total optimization time: {}".format(opt_time))
    #
    #         if timing:
    #             return x0, opt_time
    #         else:
    #             return x0

    def fast_solve(self, constraints=None, timing=False):
        '''constraints is a dictionary of constraints to pass to the scipy solver
        constraint dict must be of the form:
            constraints = {'problem' : 'lasso',
                           'bound' : t}
        where t is an np.float

        timing - bool - False don't return timings, if true then do
        '''
        sketch_function = self.sketch_type
        itr_count = 0
        ATy = self.data.T@self.targets
        # print("ATy shape {}".format(ATy.shape))
        x0 = np.zeros(shape=(self.data.shape[1],))

        if constraints is None:
            # Setup
            #ATy = np.ravel(self.data.T@self.targets)

            #covariance_matrix = self.data.T@self.data
            #summaries = self.generate_summaries()

            norm_diff = 1.0
            old_norm = 1.0

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
            # print(type(A))
            y = self.targets
            x0 = np.zeros(shape=(self.d,))
            m = int(self.sketch_dimension)
            ATy = A.T@y
            # print("In loop shape {}".format(ATy.shape))
            # print("mat-vec shape {}".format((A@x0).shape))
            setup_time = default_timer() - setup_time_start


            old_norm = 1.0
            norm_diff = 1.0
            approx_norm_sum = 0
            old_obj_val = lasso(A,x0,y,lasso_bound)
            # measurable timing vars
            opt_time = 0
            sketch_time = 0
            # sketch_time = 0 unnecessary as will come from the previous sketch call


            print("USING {} iterations".format(self.number_iterations))
            for n_iter in range(self.number_iterations):
                if norm_diff > 10E-16:
                    itr_count += 1
                    print("ITERATION {} testing sketch {}".format(itr_count, sketch_function)) # change this when using norm diff comparison

                    # Potentially remove this for a single call to sketch from the sketch_function
                    # start_sketch_time = default_timer()
                    #S_A = summaries[:,:, n_iter]
                    if sketch_function is "CountSketch":

                        if self.d < 400:
                            sketch_start = default_timer()
                            S_A = _countSketch_fast(self.nonzero_rows,self.nonzero_cols,self.nonzero_data, self.n, self.d, self.sketch_dimension)
                            sketch_time += default_timer() - sketch_start
                        else:
                            sketch_start = default_timer()
                            S_A = countSketch_dense(self.data, self.sketch_dimension)
                            sketch_time += default_timer() - sketch_start
                    elif sketch_function is "SRHT":
                        sketch_start = default_timer()
                        S_A = srht_transform1(self.data, self.sketch_dimension)
                        sketch_time += default_timer() - sketch_start


                    ### optimization ###
                    #opt_start = default_timer()
                    #sub_prob = lasso_qp_iters(S_A, ATy, covariance_mat, _lambda, x0)
                    sub_prob, end_opt_time, grad_norm = ihs_lasso_solver(S_A, ATy, A, lasso_bound, x0,timing=True)
                    #end_opt_time = default_timer()-opt_start
                    opt_time += end_opt_time
                    print("Norm of gradient: {}".format(grad_norm))


                    ### Norm checking and updates ###
                    #x_out = sub_prob[2]
                    x_out = sub_prob[0]

                    x_new = x_out[self.d:] - x_out[:self.d]
                    print("Norm of new error summand: {}".format(np.linalg.norm(x_new)))
                    updated_grad = np.linalg.norm(A.T@(A@x_new) - ATy)
                    print("Updated gradient: {}".format(updated_grad))
                    #print("iterative norm ",np.linalg.norm(x_new))
                    #new_obj_val = lasso(A,x_new,y,lasso_bound)

                    #norm_change = np.linalg.norm()
                    #print("soln norm ", np.linalg.norm(x_new,1))
                    #approx_norm_sum += np.linalg.norm(x_new,1)
                    #print("Norm diff {}".format(norm_diff))
                    x_test_approx = x0.copy() # test diference between old x0 --. now x_test_approx and new x0
                    x0 += x_new
                    new_norm = np.linalg.norm(x0,ord=2)**2
                    norm_diff = np.abs(new_norm - old_norm)/old_norm
                    old_norm = new_norm
                    #print("new_x_approx: {}".format(x0))
                    print("Norm diff: {}".format(norm_diff))


        if timing is True:
            print("returning time")
            return (x0, setup_time, sketch_time, opt_time, itr_count)
        else:
            return x0
