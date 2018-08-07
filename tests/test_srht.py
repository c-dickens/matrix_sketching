'''
Test script to ensure the SRHT sketch is working as desired
'''
import numpy as np
import pandas as pd
import unittest
import sys
sys.path.append("..")
import lib
from lib import SRHT
#from lib.srht import hadamard_transform



#################
random_seed = 10
np.random.seed(random_seed)
dir = '..'

rawdata_mat = np.load(dir + '/data/YearPredictionMSD.npy')

subset_size = 10000
X = rawdata_mat[:subset_size, 1:]
y = rawdata_mat[:subset_size, 0]
y = y[:,None]
print("Shape of data: {}".format(rawdata_mat.shape))
print("Shape of testing data: {}".format(X.shape))
print("Shape of test vector: {}".format(y.shape))

# True quantities
print("Computing true values")
covariance_matrix = X.T@X
covariance_matrix_norm = np.linalg.norm(covariance_matrix, ord='fro')
XTy_mat = X.T@y
XTy_mat_norm = np.linalg.norm(XTy_mat, ord='fro')
print("Done.")

class TestSketch(unittest.TestCase):

    # def test_srht_transform(self):
    #
    #     # Real FFT applied to columns
    #     sketch = hadamard_transform(X)
    #     #SRHT(data=X, sketch_dimension=sketch_size,second_data=np.ones_like(X))
    #     #sketch = summary.sketch(X)
    #     true_norm = np.linalg.norm(X, ord='fro')**2
    #     sketch_norm = np.linalg.norm(sketch, ord='fro')**2
    #     error = np.abs(true_norm - sketch_norm)/true_norm
    #
    #     print('SRHT difference in squared norm is {}'.format(error))
    #     self.assertTrue(error < 0.0001)
    #
    #     # Test orthogonality - need X^X approx (SX)^T (SX)
    #
    #
    #
    #     # Real FFT applied to rows
    #     # The real FFT matrix F is orthogonal,
    #     # thus (X*F) * (X*F)^T should be equal to X * X^T;
    #     # otherwise real FFT is wrong
    #     covariance_matrix_approx = sketch.T@sketch
    #     orthogonal_error = np.linalg.norm(covariance_matrix_approx - covariance_matrix, ord='fro') / covariance_matrix_norm
    #     print('Orthogonal product error: {}'.format(orthogonal_error))
    #     self.assertTrue(orthogonal_error < 0.0001)

    def test_size(self):
        sketch_size = 40

        # Check dimensionality
        print("Checking dimensions with no extra arguments")
        # No extra arguments
        summary = SRHT(data=X, sketch_dimension=sketch_size)
        sketch = summary.sketch(X)
        self.assertEqual(sketch.shape[1], X.shape[1]) # columns preserved
        self.assertEqual(sketch.shape[0],sketch_size) # rows are sketch size
        print("Passed")

        # random states added
        print("Checking with random state")
        summary = SRHT(data=X, sketch_dimension=sketch_size,random_state=random_seed)
        sketch = summary.sketch(X)
        self.assertEqual(sketch.shape[1], X.shape[1]) # columns preserved
        self.assertEqual(sketch.shape[0],sketch_size) # rows are sketch size
        print("Passed")

        # Only second_data
        print("Checking with second_data")
        summary = SRHT(data=X, sketch_dimension=sketch_size,second_data=np.ones_like(X))
        sketch = summary.sketch(X)
        self.assertEqual(sketch.shape[1], X.shape[1]) # columns preserved
        self.assertEqual(sketch.shape[0],sketch_size) # rows are sketch size
        print("Passed")

        # Check with both random_state and second_data
        print("Checking with both arguments")
        summary = SRHT(data=X, sketch_dimension=sketch_size,random_state=random_seed, second_data=np.ones_like(X))
        sketch = summary.sketch(X)
        self.assertEqual(sketch.shape[1], X.shape[1]) # columns preserved
        self.assertEqual(sketch.shape[0],sketch_size) # rows are sketch size
        print("Passed")


    def test_multiply_error(self):
        '''
        Test the gaussian sketching function from the Gaussian class
        Test over num_trials for averaging and check that increasing the
        sketch size decreases the error
        '''

        num_trials = 10
        sketch_size_1 = 100
        sketch_size_2 = 300
        sketch_size_3 = 1000
        error1, error2, error3 = 0,0,0

        for trial in range(num_trials):
            summary = SRHT(data=X,
                                     sketch_dimension=sketch_size_1)
            sketch = summary.sketch(X)
            error1 += np.linalg.norm(covariance_matrix - sketch.T@sketch, ord='fro') / covariance_matrix_norm
        error1 /= num_trials

        for trial in range(num_trials):
            summary = SRHT(data=X,
                                     sketch_dimension=sketch_size_2)
            sketch = summary.sketch(X)
            error2 += np.linalg.norm(covariance_matrix - sketch.T@sketch, ord='fro') / covariance_matrix_norm
        error2 /= num_trials

        for trial in range(num_trials):
            summary = SRHT(data=X,
                                     sketch_dimension=sketch_size_3)
            sketch = summary.sketch(X)
            error3 += np.linalg.norm(covariance_matrix - sketch.T@sketch, ord='fro') / covariance_matrix_norm
        error3 /= num_trials

        print('Approximation error for sketch size = {}: {}'.format(sketch_size_1,error1))
        print('Approximation error for sketch size = {}: {}'.format(sketch_size_2,error2))
        print('Approximation error for sketch size = {}: {}'.format(sketch_size_3,error3))
        self.assertTrue(error2 < error1)
        self.assertTrue(error3 < error2)

    #
    # def test_mat_vec_product(self):
    #     '''Tests that the matrix vector product is preserved'''
    #     num_trials = 10
    #     sketch_size_1 = 100
    #     sketch_size_2 = 300
    #     sketch_size_3 = 1000
    #     error1, error2, error3 = 0,0,0
    #
    #     #Xy = np.concatenate((X,y), axis=1)
    #
    #     for trial in range(num_trials):
    #         summary = SRHT(data=X,
    #                                  sketch_dimension=sketch_size_1,
    #                                  second_data = y)
    #         S_X, S_y = summary.sketch_product(X,y)
    #
    #         error1 += np.linalg.norm(XTy_mat - S_X.T@S_y, ord='fro') / XTy_mat_norm
    #     error1 /= num_trials
    #
    #     for trial in range(num_trials):
    #         summary = SRHT(data=X,
    #                                  sketch_dimension=sketch_size_2,
    #                                  second_data = y)
    #         S_X, S_y = summary.sketch_product(X,y)
    #
    #         error2 += np.linalg.norm(XTy_mat - S_X.T@S_y, ord='fro') / XTy_mat_norm
    #
    #     error2 /= num_trials
    #
    #     for trial in range(num_trials):
    #         summary = SRHT(data=X,
    #                                  sketch_dimension=sketch_size_3,
    #                                  second_data = y)
    #         S_X, S_y = summary.sketch_product(X,y)
    #
    #         error3 += np.linalg.norm(XTy_mat - S_X.T@S_y, ord='fro') / XTy_mat_norm
    #     error3 /= num_trials
    #
    #     print('Approximation error for X^Ty with sketch size = {}: {}'.format(sketch_size_1,error1))
    #     print('Approximation error for X^Ty with sketch size = {}: {}'.format(sketch_size_2,error2))
    #     print('Approximation error for X^Ty with sketch size = {}: {}'.format(sketch_size_3,error3))
    #     self.assertTrue(error2 < error1)
    #     self.assertTrue(error3 < error2)


if __name__ == "__main__":
    unittest.main()
