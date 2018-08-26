'''
Synthetic data generators
'''
import numpy as np
from scipy.sparse import random
from scipy.sparse import coo_matrix
from sklearn.preprocessing import StandardScaler




def unconstrained_regession_data(nsamples, nfeatures, variance):#, random_seed=100):
    '''
    Generate data as described in https://arxiv.org/pdf/1411.0347.pdf 3.1
    1. Generate A in R^{n \times d} with A_ij inn N(0,1)
    2. Choose x^* from S^{d-1}
    3. Set y = Ax^* + w where w ~ N(0,variance*I)
    '''
    #np.random.seed(random_seed)
    A = np.random.randn(nsamples, nfeatures)
    x_true = np.random.randn(nfeatures)
    x_true /= np.linalg.norm(x_true)
    noise = np.random.normal(loc=0.0,scale=variance,size=(nsamples,))
    y = A@x_true + noise
    return A, y, x_true

def generate_lasso_data(m, n, data_density=0.1, sigma=5, sol_density=0.2):
    "Generates data matrix X and observations Y."
    np.random.seed(1)
    beta_star = np.random.randn(n)
    idxs = np.random.choice(range(n), int((1-sol_density)*n), replace=False)
    for idx in idxs:
        beta_star[idx] = 0
    X = random(m,n,data_density)#np.random.randn(m,n)
    Y = X.dot(beta_star) + np.random.normal(0, sigma, size=m)
    scaler = StandardScaler(with_mean=False).fit(X)
    sparse_X = scaler.transform(X)
    dense_X = sparse_X.toarray()
    sparse_X = coo_matrix(sparse_X)

    #scale_y = Normalizer().fit(Y)
    #new_y = scaler.transform(Y)
    return sparse_X, dense_X, Y, beta_star
