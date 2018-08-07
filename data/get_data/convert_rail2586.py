'''
Script to convert the rail2586 data to
.npy format for random projection exps
'''


import pandas as pd
import numpy as np
import scipy
from scipy.io import loadmat


if __name__ == "__main__":
    inputFileName = 'rail2586.mat'
    outputFileName = '../rail2586'
    d = 2586 # number of features


    print("Reading")
    matrix = scipy.io.loadmat(inputFileName)
    print(matrix)
    data = matrix['X']
    X = data.tocoo()
    print("Shape of the dataset: {}".format(X.shape))
    print("Type of data: {}".format(type(X)))
    np.save(outputFileName, X)
