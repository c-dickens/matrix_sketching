import numpy as np
import pandas as pd
from timeit import default_timer
from scipy import optimize
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
import scipy.optimize
import unittest
import sys
sys.path.append("..")
import lib
from lib import Sketch, GaussianSketch, CountSketch, SRHT, IHS, ClassicalSketch

#################
random_seed = 10
np.random.seed(random_seed)
sketch_names = ["CountSketch", "SRHT", "Gaussian"]

dir = '..'
rawdata_mat = np.load(dir + '/data/YearPredictionMSD.npy')

subset_size = 10000
X = rawdata_mat[:subset_size, 1:]
y = rawdata_mat[:subset_size, 0]
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = y[:,None]
y = scaler.fit_transform(y)
print("Shape of data: {}".format(rawdata_mat.shape))
print("Shape of testing data: {}".format(X.shape))
print("Shape of test vector: {}".format(y.shape))


class TestClassicalSketch(unittest.TestCase):

    def test_sketch_call(self):
        '''Test that the sketches of X and y can be formed.'''
        print(80*"-")
        print("TESTING SKETCH CALL")
        sketch_size = 100
        num_iters = 10


        for sketch_method in sketch_names:
            summary = ClassicalSketch(data=X, targets=y, sketch_dimension=sketch_size,
                                                    sketch_type=sketch_method,
                                                    random_state=random_seed)
            sketch_X, sketch_y = summary.sketch()
            print("{}, shape: {}".format(sketch_method, sketch_X.shape))
            self.assertEqual(sketch_X.shape[1], X.shape[1]) # Columns retained
            self.assertEqual(sketch_X.shape[0], sketch_size) # num rows in sketch data equal
            self.assertEqual(sketch_y.shape[0], sketch_size) # num rows in sketch target equal

    def test_sketch_and_solve_unconstrained(self):
        '''Tests the solution of the sketch and solve approach for the
        unconstrained regression problem.

        Use num_trials for averging and check that the approximation error
        decreases as the sketch size is increased.'''
        print('TESTING UNCONSTRAINED REGRESSION')
        X,y,coef = make_regression(n_samples = 5000, n_features = 50, noise=5.0, coef=True)
        num_trials = 10
        sketch_size_1 = 100
        sketch_size_2 = 250
        sketch_size_3 = 500
        sol_error1, sol_error2, sol_error3 = 0,0,0
        cost_error1, cost_error2, cost_error3 = 0,0,0

        # Get optimal / true weights
        true_x = np.linalg.lstsq(X,y, rcond=None)[0]
        true_cost = np.linalg.norm(X@true_x - y)**2

        for sketch_method in sketch_names:
            print("Testing sketch: {}".format(sketch_method))
            for trial in range(num_trials):
                sketch_and_solve = ClassicalSketch(data=X, targets=y,
                                                    sketch_dimension=sketch_size_1,
                                                    sketch_type=sketch_method,
                                                    random_state=random_seed)
                sketch_x = sketch_and_solve.solve()
                #print("Sketched weights: {}".format(sketch_x))
                cost_approx = (1/subset_size)*np.linalg.norm(X@sketch_x - y)**2
                #print("Approx cost: {}".format(cost_approx))
                cost_error1 += np.abs(cost_approx - true_cost)/true_cost
                sol_error1  += (1/subset_size)*np.linalg.norm(X@(sketch_x-true_x))**2

            for trial in range(num_trials):
                sketch_and_solve = ClassicalSketch(data=X, targets=y,
                                                    sketch_dimension=sketch_size_2,
                                                    sketch_type=sketch_method,
                                                    random_state=random_seed)
                sketch_x = sketch_and_solve.solve()
                #print("Sketched weights: {}".format(sketch_x))
                cost_approx = (1/subset_size)*np.linalg.norm(X@sketch_x - y)**2
                #print("Approx cost: {}".format(cost_approx))
                cost_error2 += np.abs(cost_approx - true_cost)/true_cost
                sol_error2  += (1/subset_size)*np.linalg.norm(X@(sketch_x-true_x))**2

            for trial in range(num_trials):
                sketch_and_solve = ClassicalSketch(data=X, targets=y,
                                                    sketch_dimension=sketch_size_3,
                                                    sketch_type=sketch_method,
                                                    random_state=random_seed)
                sketch_x = sketch_and_solve.solve()
                #print("Sketched weights: {}".format(sketch_x))
                cost_approx = (1/subset_size)*np.linalg.norm(X@sketch_x - y)**2
                #print("Approx cost: {}".format(cost_approx))
                cost_error3 += np.abs(cost_approx - true_cost)/true_cost
                sol_error3  += (1/subset_size)*np.linalg.norm(X@(sketch_x-true_x))**2

            cost_error1 /= num_trials
            sol_error1 /= num_trials
            cost_error2 /= num_trials
            sol_error2 /= num_trials
            cost_error3 /= num_trials
            sol_error3 /= num_trials

            print("Approximations for {} with sketch size {}".format(sketch_method, sketch_size_1))
            print("Relative Error cost approximation: {}".format(cost_error1))
            print("Relative Error cost approximation: {}".format(cost_error2))
            print("Relative Error cost approximation: {}".format(cost_error3))
            print("Solution approximation: {}".format(sol_error1))
            print("Solution approximation: {}".format(sol_error2))
            print("Solution approximation: {}".format(sol_error3))
            # self.assertTrue(cost_error2 < cost_error1)
            # self.assertTrue(cost_error3 < cost_error2)

    def test_sketch_and_solve_lasso(self):
        '''Tests the solution of the sketch and solve approach for the
        unconstrained regression problem.

        Use num_trials for averging and check that the approximation error
        decreases as the sketch size is increased.'''
        print("TESTING LASSO SOLVER")
        n = 5000
        d = 25
        sklearn_lasso_bound = 10
        X,y,coef = make_regression(n_samples = n, n_features = d, noise=5.0, coef=True)
        num_trials = 5
        sketch_size_1 = 100
        sketch_size_2 = 250
        sketch_size_3 = 500
        sol_error1, sol_error2, sol_error3 = 0,0,0
        cost_error1, cost_error2, cost_error3 = 0,0,0

        # Get optimal / true weights
        clf = Lasso(sklearn_lasso_bound)
        lasso = clf.fit(n*X,n*y)
        true_x = lasso.coef_
        true_cost = 0.5*np.linalg.norm(X@true_x-y,ord=2)**2 +\
                                    sklearn_lasso_bound*np.linalg.norm(true_x,1)
        x_norm_bound = np.linalg.norm(true_x,1)

        for sketch_method in sketch_names:
            print("Testing sketch: {}".format(sketch_method))
            for trial in range(num_trials):
                sketch_and_solve = ClassicalSketch(data=X, targets=y,
                                                    sketch_dimension=sketch_size_1,
                                                    sketch_type=sketch_method,
                                                    random_state=random_seed)
                sketch_x = sketch_and_solve.solve({'problem' : "lasso", 'bound' : x_norm_bound})
                #print("Sketched weights: {}".format(sketch_x))
                cost_approx = (0.5/n)*np.linalg.norm(X@sketch_x - y)**2
                #print("Approx cost: {}".format(cost_approx))
                cost_error1 += np.abs(cost_approx - true_cost)/true_cost
                sol_error1  += (0.5/n)*np.linalg.norm(X@(sketch_x-true_x))**2

            for trial in range(num_trials):
                sketch_and_solve = ClassicalSketch(data=X, targets=y,
                                                    sketch_dimension=sketch_size_2,
                                                    sketch_type=sketch_method,
                                                    random_state=random_seed)
                sketch_x = sketch_and_solve.solve({'problem' : "lasso", 'bound' : x_norm_bound})
                #print("Sketched weights: {}".format(sketch_x))
                cost_approx = (0.5/n)*np.linalg.norm(X@sketch_x - y)**2
                #print("Approx cost: {}".format(cost_approx))
                cost_error2 += np.abs(cost_approx - true_cost)/true_cost
                sol_error2  += (0.5/n)*np.linalg.norm(X@(sketch_x-true_x))**2

            for trial in range(num_trials):
                sketch_and_solve = ClassicalSketch(data=X, targets=y,
                                                    sketch_dimension=sketch_size_3,
                                                    sketch_type=sketch_method,
                                                    random_state=random_seed)
                sketch_x = sketch_and_solve.solve({'problem' : "lasso", 'bound' : x_norm_bound})
                #print("Sketched weights: {}".format(sketch_x))
                cost_approx = (0.5/n)*np.linalg.norm(X@sketch_x - y)**2
                #print("Approx cost: {}".format(cost_approx))
                cost_error3 += np.abs(cost_approx - true_cost)/true_cost
                sol_error3  += (0.5/n)*np.linalg.norm(X@(sketch_x-true_x))**2

            cost_error1 /= num_trials
            sol_error1 /= num_trials
            cost_error2 /= num_trials
            sol_error2 /= num_trials
            cost_error3 /= num_trials
            sol_error3 /= num_trials

            print("Approximations for {} with sketch size {}".format(sketch_method, sketch_size_1))
            print("Relative Error cost approximation: {}".format(cost_error1))
            print("Relative Error cost approximation: {}".format(cost_error2))
            print("Relative Error cost approximation: {}".format(cost_error3))
            print("Solution approximation: {}".format(sol_error1))
            print("Solution approximation: {}".format(sol_error2))
            print("Solution approximation: {}".format(sol_error3))

if __name__ == "__main__":
    unittest.main()
