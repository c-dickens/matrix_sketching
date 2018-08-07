'''
CountSketch sketching algorithm

CountSketch memory is marginally faster but the performance is generally the same.
'''

import numpy as np
import numba
from numba import jit
from . import Sketch
from scipy.sparse import coo_matrix
import scipy.sparse
from timeit import default_timer

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
def countSketch_dense(data, sketch_dimension):
    '''
    count sketch for dense data.
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

@jit('float64[:,:](int32[:],int32[:],float64[:],int64,int64,int64)',nopython=True)
def _countSketch_fast(nonzero_rows, nonzero_cols, nonzero_data, n, d, sketch_dimension):
    '''Perform count sketch on preprocessed data'''
    sketch = np.zeros((sketch_dimension, d))
    hashedIndices = np.random.choice(sketch_dimension, n, replace=True)
    randSigns = np.random.choice(2, n, replace=True) * 2 - 1
    nnz_id = 0
    old_row = 0
    bucket, sign = hashedIndices[0], randSigns[0]

    for row_id in nonzero_rows:
        col_id = nonzero_cols[nnz_id]
        data_val = nonzero_data[nnz_id]
        if row_id == old_row:
            sketch[bucket, col_id] += sign*data_val
        else:
            bucket = hashedIndices[row_id]
            sign = randSigns[row_id]
            sketch[bucket, col_id] += sign*data_val
            old_row = row_id
        nnz_id += 1
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

    Needs to have the inputs sparse matrix as a coo_matrix so that the
    .row/.col/.data attributes can be used.
    '''

    def __init__(self, data, sketch_dimension, random_state=None, second_data=None, timing=False):
        if random_state is not None or second_data is not None:
            super(CountSketch,self).__init__(data, sketch_dimension,\
                                                        random_state, second_data)
        else:
            super(CountSketch,self).__init__(data, sketch_dimension)

        self.n, self.d = data.shape

        if scipy.sparse.issparse(data):
            self.nonzero_rows = data.row
            self.nonzero_cols = data.col
            self.nonzero_data = data.data
            self.sparse_bool = True
        else:
            X_coo = coo_matrix(self.data)
            self.nonzero_rows = X_coo.row
            self.nonzero_cols = X_coo.col
            self.nonzero_data = X_coo.data
            self.sparse_bool = False
        self.timing = timing

    def sketch(self, data):
        '''data argument superfluous but indicates that sketching is done on X
        and maintains consistency'''
        # if self.timing:
        #     start = default_timer()
        #     summary = _countSketch_fast(self.nonzero_rows, self.nonzero_cols, self.nonzero_data, self.n, self.d, self.sketch_dimension)
        #     sketch_time = default_timer() - start
        #     return summary, sketch_time
        # else:
        #     summary = _countSketch_fast(self.nonzero_rows, self.nonzero_cols, self.nonzero_data, self.n, self.d, self.sketch_dimension)
        #     return summary
        #if self.sparse_bool:
        summary = _countSketch_fast(self.nonzero_rows, self.nonzero_cols, self.nonzero_data, self.n, self.d, self.sketch_dimension)
        return summary
        # else:
        #     print("Using dense method")
        #     summary = countSketch_dense(data, self.sketch_dimension)
        #     return summary

    #
    # def sketch_memory(self,data):
    #     summary = countSketch_memory(data, self.sketch_dimension)
    #     return summary

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
