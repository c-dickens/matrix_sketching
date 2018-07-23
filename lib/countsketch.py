'''
CountSketch sketching algorithm

CountSketch memory is marginally faster but the performance is generally the same.
'''

import numpy as np
import numba
from numba import jit
from . import Sketch

@jit(nopython=True)
def _countSketch(data, sketch_dimension):
    '''Sketching as a helper function for the class function call with jit'''
    n,d = data.shape
    sketch = np.zeros((sketch_dimension,d))
    nonzero_rows, nonzero_cols = np.nonzero(data)
    hashedIndices = np.random.choice(sketch_dimension, n, replace=True)
    randSigns = np.random.choice(2, n, replace=True) * 2 - 1
    for ii,jj in zip(nonzero_rows,nonzero_cols):
        bucket = hashedIndices[ii]
        sketch[bucket, jj] += randSigns[ii]*data[ii,jj]
    return sketch

@jit(nopython=True)
def countSketch_memory(data, sketch_dimension):
    '''
    An in-memory implementation of the count sketch.
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

    return sketch

class CountSketch(Sketch):
    '''Reduces dimensionality by using a CountSketch random projection in the
    streaming model.

    Parameters
    ----------
    sketch_dimension: int - number of rows to compress into.

    References
    -----------
    https://arxiv.org/abs/1411.4357 -- DP Woodruff
    '''

    def __init__(self, data, sketch_dimension, random_state=None, second_data=None):
        if random_state is not None or second_data is not None:
            super(CountSketch,self).__init__(data, sketch_dimension,\
                                                        random_state, second_data)
        else:
            super(CountSketch,self).__init__(data, sketch_dimension)

    def sketch(self, data):
        summary = _countSketch(data, self.sketch_dimension)
        return summary

    def sketch_memory(self,data):
        summary = countSketch_memory(data, self.sketch_dimension)
        return summary

    # def sketch_product(self, first_data, second_data):
    #     '''
    #     sketches self and another dataset.
    #     Generates a sketch of the identity to act as the same sketching matrix
    #     on both datasets to pass the test.
    #     '''
    #     pass
    #     if self.random_state is None:
    #         pass
        #sketch_X = self.sketch(X)
        #I = np.identity(first_data.shape[0])
        #S = self.sketch(I)
        #S_X = S@first_data
        #S_Y = S@second_data

        #return S_X, S_Y
