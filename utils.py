'''utility functions used throughout'''

import numpy as np
import scipy as sp
from numba import jit
from hadamard import fastwht
from sklearn.linear_model import Lasso
from timeit import default_timer

## Typical error measurement functions

def mean_square_error(x1, x2):
    '''compute ||x2 - x1||_2^2'''
    return np.linalg.norm(x2-x1)**2

def prediction_error(data,x1,x2):
    '''compute np.sqrt(1/n)*||A(x1-x2)||_2'''
    return (1/np.sqrt(data.shape[0]))*np.linalg.norm(data@(x1-x2))


### Lasso functions
def original_lasso_objective(X,y, regulariser,x, penalty=False):

    if penalty:
        return 0.5*np.linalg.norm(X@x-y,ord=2)**2 + regulariser*np.linalg.norm(x,1)
    else:
        return 0.5*np.linalg.norm(X@x-y,ord=2)**2

def sklearn_wrapper(X,y,n,d, regulariser, trials):
    '''solves the lasso problem in the sklearn sense by dividing out number
    of rows for normalisation.'''

    clf = Lasso(regulariser)
    lasso_time = 0
    for i in range(trials):
        print("Trial ", i)
        lasso_start = default_timer()
        lasso = clf.fit(np.sqrt(n)*X,np.sqrt(n)*y)
        lasso_time += default_timer() - lasso_start
        x_opt = lasso.coef_
        f_opt = original_lasso_objective(X,y,regulariser,x_opt,penalty=True)
    return x_opt, f_opt, lasso_time/trials





### sketches
