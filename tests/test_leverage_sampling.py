'''Tests the countsketch.py methods'''

import numpy as np
import pandas as pd
import unittest
import sys
sys.path.append("..")
import lib
from lib import Sketch
from lib import LeverageScoreSampler



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
covariance_matrix = X.T@X
covariance_matrix_norm = np.linalg.norm(covariance_matrix, ord='fro')
XTy_mat = X.T@y
XTy_mat_norm = np.linalg.norm(XTy_mat, ord='fro')

class TestSketch(unittest.TestCase):

    def test_size(self):
        sketch_size = 40

        # Check dimensionality
        print("Checking dimensions with no extra arguments")
        # No extra arguments
        summary = LeverageScoreSampler(data=X, sketch_dimension=sketch_size)

        sketch = summary.sketch(X)
        self.assertEqual(sketch.shape[1], X.shape[1]) # columns preserved
        self.assertEqual(sketch.shape[0],sketch_size) # rows are sketch size
        print("Passed")

        # random states added
        print("Checking with random state")
        summary = LeverageScoreSampler(data=X, sketch_dimension=sketch_size,random_state=random_seed)
        sketch = summary.sketch(X)
        self.assertEqual(sketch.shape[1], X.shape[1]) # columns preserved
        self.assertEqual(sketch.shape[0],sketch_size) # rows are sketch size
        print("Passed")

        # Only second_data
        print("Checking with second_data")
        summary = LeverageScoreSampler(data=X, sketch_dimension=sketch_size,second_data=np.ones_like(X))
        sketch = summary.sketch(X)
        self.assertEqual(sketch.shape[1], X.shape[1]) # columns preserved
        self.assertEqual(sketch.shape[0],sketch_size) # rows are sketch size
        print("Passed")

        # Check with both random_state and second_data
        print("Checking with both arguments")
        summary = LeverageScoreSampler(data=X, sketch_dimension=sketch_size,random_state=random_seed, second_data=np.ones_like(X))
        sketch = summary.sketch(X)
        self.assertEqual(sketch.shape[1], X.shape[1]) # columns preserved
        self.assertEqual(sketch.shape[0],sketch_size) # rows are sketch size
        print("Passed")


    def test_exact_leverage_scores(self):
        '''Checks that the leverage scores are correct'''
        sketch_size = 40
        print("Getting exact leverage scores.")
        summary = LeverageScoreSampler(data=X, sketch_dimension=sketch_size,random_state=random_seed, second_data=np.ones_like(X))
        leverage_scores = summary.get_leverage_scores()
        print("Got leverage scores.")
        print("Exact leverage scores:\nmax = {}, min = {}".format(np.max(leverage_scores), np.min(leverage_scores)))
        self.assertTrue(np.max(leverage_scores) >= 0)
        self.assertTrue(np.max(leverage_scores) <= 1)
        self.assertEqual(leverage_scores.shape[0], X.shape[0])

if __name__ == '__main__':
    unittest.main()
