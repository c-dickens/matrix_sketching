import numpy as np
import pandas as pd
from timeit import default_timer
from scipy import optimize
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import quadprog as qp
import scipy.optimize
import unittest
import sys
sys.path.append("..")
import lib
from lib import Sketch,  CountSketch, SRHT, IHS

#################
random_seed = 10
np.random.seed(random_seed)
sketch_names = ["CountSketch", "SRHT"]

dir = '..'
rawdata_mat = np.load(dir + '/data/YearPredictionMSD.npy')

subset_size = 1000
X = rawdata_mat[:subset_size, 1:]
y = rawdata_mat[:subset_size, 0]
y = y[:,None]
print("Shape of data: {}".format(rawdata_mat.shape))
print("Shape of testing data: {}".format(X.shape))
print("Shape of test vector: {}".format(y.shape))



########## LASSO SCIPY FUNCTION ############
### QP approaches
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


def generate_lasso_data(m, n, sigma=5, density=0.2):
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


class TestSketch(unittest.TestCase):

    def test_sketch_call(self):
        print(80*"-")
        print("TESTING SKETCH CALL")
        sketch_size = 100
        num_iters = 10


        for sketch_method in sketch_names:
            print("Testing {}".format(sketch_method))
            summary = IHS(data=X, targets=y, sketch_dimension=sketch_size,
                                                    sketch_type=sketch_method,
                                                    number_iterations=num_iters,
                                                    random_state=random_seed)
            sketch = summary.sketch(X)
            print("{}, shape: {}".format(sketch_method, sketch.shape))
            self.assertEqual(sketch.shape[1], X.shape[1])
            self.assertEqual(sketch.shape[0], sketch_size)

    def test_summary_generation(self):
        sketch_size = 100
        num_iters = 10
        print("TESTING SUMMARY GENERATION FUNCTION")

        for sketch_method in sketch_names:
            iterative_hessian = IHS(data=X, targets=y, sketch_dimension=sketch_size,
                                                    sketch_type=sketch_method,
                                                    number_iterations=num_iters,
                                                    random_state=random_seed)
            print("Generating summaries")
            all_sketches = iterative_hessian.generate_summaries()
            print("Shape of all summaries: {}".format(all_sketches.shape))

            # Num summaries is equal to number of iterations
            self.assertEqual(all_sketches.shape[2], num_iters)
        print("COMPLETED TESTING SUMMARY GENERATION FUNCTION")

    def test_unconstrained_regression(self):
        '''Show that a random regression instance is approximated by the
        hessian sketching scheme'''

        print("TESTING UNCONSTRAINED ITERATIVE HESSIAN SKETCH ALGORITHM")
        sketch_size = 500
        num_iters = 10
        # Setup
        syn_data,syn_targets,coef = make_regression(n_samples = 5000, n_features = 2, coef=True)
        optimal_weights = np.linalg.lstsq(syn_data,syn_targets)[0]
        for sketch_method in sketch_names:
            iterative_hessian = IHS(data=syn_data, targets=syn_targets, sketch_dimension=sketch_size,
                                                    sketch_type=sketch_method,
                                                    number_iterations=num_iters,
                                                    random_state=random_seed)
            print("STARTING IHS ALGORITHM WITH {}".format(sketch_method), 60*"*")
            #start = default_timer()
            x_approx = iterative_hessian.solve()
            #ihs_time = default_timer() - start
            #print("Alg took: {}s".format(ihs_time))
            print("DONE IHS ALG WITH {}".format(sketch_method), 60*"*")
            #print("x shape: {}".format(x_approx.shape))
            #print("data shape: {}".format(syn_data.shape))
            #print("Weights shape: {}".format(optimal_weights.shape))
            print("Sketch: {}".format(sketch_method))
            #print("Approx. weights: {}".format(x_approx))
            #print("Optimal weights: {}".format(optimal_weights))
            print("||x^* - x'||_A^2: {}".format((np.linalg.norm(syn_data@(x_approx - optimal_weights)**2/syn_data.shape[0]))))
            np.testing.assert_allclose(x_approx, optimal_weights)
        print("COMPLETED UNCONSTRAINED ITERATIVE HESSIAN SKETCH ALGORITHM")


    def test_lasso_regression(self):
        '''Show that a random lasso instance is approximated by the
        hessian sketching scheme'''
        print(80*"-")
        print("TESTING LASSO ITERATIVE HESSIAN SKETCH ALGORITHM")

        ncols = 250
        nrows = 10000
        sketch_size = 1000
        sklearn_lasso_bound = 1000
        X, y, coef = generate_lasso_data(nrows, ncols, sigma=10, density=0.5)
        trials = 5

        ### Test Sklearn implementation
        clf = Lasso(sklearn_lasso_bound)
        x_opt = clf.fit((X.shape[0])*X,(X.shape[0])*y).coef_
        print("Potential norm bound for ihs: {}".format(np.linalg.norm(x_opt,1)))
        ihs_lasso_bound = np.linalg.norm(x_opt,1) + 1E-5
        ### Test QP formulation with quadprog
        result = qp_lasso(X, y, ihs_lasso_bound)
        x = result[0]
        x_qp = -1.0*(x[ncols:] - x[:ncols])
        #print("x opt ",x_opt)
        #print("QP x", -1.0*x_qp)
        np.testing.assert_almost_equal(np.linalg.norm(x_opt,1), np.linalg.norm(x_qp,1), decimal=1)
        print("QP SOLVER AND SKLEARN HAVE CORRESPONDING SOLUTIONS")


        for sketch_method in sketch_names:
            ihs_lasso = IHS(data=X, targets=y, sketch_dimension=sketch_size,
                                                    sketch_type=sketch_method,
                                                    number_iterations=20,
                                                    random_state=random_seed)
            print("STARTING IHS-LASSO ALGORITHM WITH {}".format(sketch_method), 60*"*")
            #start = default_timer()
            x_ihs = ihs_lasso.solve({'problem' : "lasso", 'bound' : ihs_lasso_bound})
            print("Comparing difference between opt and approx:")
            #print(np.linalg.norm(x_opt - x_ihs))
            print("||x^* - x'||_A^2: {}".format((np.linalg.norm(X@(x_opt - x_ihs)**2/X.shape[0]))))

            # test that the constrain bound is met
            self.assertTrue(np.linalg.norm(x_ihs,1) - ihs_lasso_bound < 0.01)

            # Test convergence
            np.testing.assert_array_almost_equal(x_opt, x_ihs, decimal=4)
            print("SOLUTION IS CORRECT")



if __name__ == "__main__":
    unittest.main()
