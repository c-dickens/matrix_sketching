import numpy as np
import pandas as pd
from timeit import default_timer
from scipy import optimize
from sklearn.datasets import make_regression
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
def lasso(x, A, b):
    return 0.5*np.linalg.norm(A@x - b)**2 #+ _lambda*np.sum(np.abs(x))


class TestSketch(unittest.TestCase):

    def test_sketch_call(self):
        print(80*"-")
        print("TESTING SKETCH CALL")
        sketch_size = 100
        num_iters = 10


        for sketch_method in sketch_names:
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

        print("TESTING UNCONSTRINAED ITERATIVE HESSIAN SKETCH ALGORITHM")
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
        sketch_size = 5000
        num_iters = 1

        lasso_bound = 50.0

        #  For the quadprog solver we use.
        constraints = {'type' : 'lasso',
                       'bound' : lasso_bound}


        # For scipy solver to check quadprog output
        cons = ({'type' : 'ineq',
                 #'fun'  : lambda x: np.sum(np.abs(x)) <= lasso_bound}) need differentiable constraints
                 'fun' : lambda x : lasso_bound - np.sum(x)**2})
        # Setup
        syn_data,syn_targets,coef = make_regression(n_samples = 5000, n_features = 2, coef=True)
        syn_data = syn_data.astype(np.float32)
        syn_targets = syn_targets.astype(np.float32)

        # Scipy Lasso solver
        print("Entering scipy solver")
        # scipy_result = optimize.minimize(lasso,np.zeros(syn_data.shape[1]),
        #                                 args = (syn_data,syn_targets, lasso_bound),
        #                                 method='SLSQP')
        scipy_result = optimize.minimize(lasso,np.zeros(syn_data.shape[1],dtype=np.float32),
                                         args=(syn_data,syn_targets),
                                         constraints=cons)
        print("scipy optimisation complete.")
        optimal_weights = scipy_result['x']
        print("Optimal x {}".format(optimal_weights))
        print("Entering IHS algorithm.")
        for sketch_method in sketch_names:
            iterative_hessian = IHS(data=syn_data, targets=syn_targets, sketch_dimension=sketch_size,
                                                    sketch_type=sketch_method,
                                                    number_iterations=num_iters,
                                                    random_state=random_seed)
            print("STARTING IHS ALGORITHM WITH {}".format(sketch_method))
            #start = default_timer()
            x_approx = iterative_hessian.solve(constraints)
            #ihs_time = default_timer() - start
            #print("Alg took: {}s".format(ihs_time))
            print("DONE IHS ALG WITH {}".format(sketch_method))
            #print("x shape: {}".format(x_approx.shape))
            #print("data shape: {}".format(syn_data.shape))
            #print("Weights shape: {}".format(optimal_weights.shape))
            print("Sketch: {}".format(sketch_method))
            print("Approx. weights: {}".format(x_approx))
            #print("Optimal weights: {}".format(optimal_weights))
            #print("||x^* - x'||_A^2: {}".format((np.linalg.norm(syn_data@(x_approx - optimal_weights)**2/syn_data.shape[0]))))
            #p.testing.assert_allclose(x_approx, optimal_weights)
        print("COMPLETED LASSO ITERATIVE HESSIAN SKETCH ALGORITHM")


if __name__ == "__main__":
    unittest.main()
