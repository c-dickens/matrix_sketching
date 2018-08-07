'''
Script to get data from UCI ML repo and convert to
.npy format for random projection exps
'''


import pandas as pd
import numpy as np

def get_data():
    '''Pulls the data from the UCI ML repo'''



if __name__ == "__main__":
    inputFileName = 'YearPredictionMSD.txt'
    outputFileName = '../YearPredictionMSD'
    d = 90 # number of features
    #processLibSVMData(inputFileName, d)
    # print("Reading")
    # matrix = np.loadtxt(inputFileName, dtype=np.float, delimiter=',')
    # matrix = matrix.astype(np.float)
    # print("Matrix has shape {}".format(matrix.shape))
    # print("Saving")
    # np.save(inputFileName, matrix)

    print("Downloading")
    file = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip'
    #file = "https://sparse.tamu.edu/MM/Mittelmann/rail2586.tar.gz"

    df = pd.read_csv(file)
    print("Data shape {}".format(df.shape))

    data = df.values
    X = data[:,1:]
    y = data[:,0]
    #print("Data shape is {}".format(X.shape))
    #print("Target shape {}".format(y.shape))
    #tidy_data = np.hstack((X,y),axis=1)
    np.save(outputFileName, data)
