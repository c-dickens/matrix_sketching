'''
Class instantiation of an efficient SRHT sketch.
using library https://bitbucket.org/vegarant/fastwht

References
-------------
[1] - https://arxiv.org/abs/1411.4357
[2] - https://github.com/EMS-TU-Ilmenau/fastmat
[3] - https://github.com/nbarbey/fht
'''

import numpy as np
from hadamard import fastwht
from timeit import default_timer
from . import Sketch
import scipy as sp
import numba
from numba import jit

def shift_bit_length(x):
    '''Given int x find next largest power of 2.
    If x is a power of 2 then x is returned '''
    return 1<<(x-1).bit_length()

def srht_transform1(input_matrix, sketch_size, seed=None):
    '''Generate a sketching matrix S to reduce the sample count (i.e. sketching
    on the LHS of data) via the Subsampled Hadamard Transform.

    Given an input_matrix ``A`` of size ``(n, d)``, compute a matrix ``A'`` of
    size (sketch_size, d) which holds:
    .. math:: ||Ax|| = (1 \pm \epsilon)||A'x||
    with high probability. [1]
    The error is related to the number of rows of the sketch and it is bounded

    tbc


    Parameters
    ----------
    input_matrix: array_like
        Input matrix, of shape ``(n, d)``.
    sketch_size: int
        Number of rows for the sketch.
    seed : None or int or `numpy.random.RandomState` instance, optional
        This parameter defines the ``RandomState`` object to use for drawing
        random variates.
        If None (or ``np.random``), the global ``np.random`` state is used.
        If integer, it is used to seed the local ``RandomState`` instance.
        Default is None.
    Returns
    -------
    S_A : array_like
        Sketch of the input matrix ``A``, of size ``(sketch_size, d)``.

    Notes
    -------
    This implementation of the SRHT is fast up to the line which requires the
    copying of the fastmat PRODUCT type to a NumPy array [2].
    The fastmat library is used to quickly compute D*A and the fht library is
    used to compute the product H(DA) quickly by exploiting the FFT [3].

    References
    -------------
    [1] - https://arxiv.org/abs/1411.4357
    [2] - https://github.com/EMS-TU-Ilmenau/fastmat
    [3] - https://github.com/nbarbey/fht



    '''
    nrows = input_matrix.shape[0]
    diag = np.random.choice([1,-1], nrows)
    diag = diag[:,None]
    # print("diag shape: {}".format(diag.shape))
    # print("input mat shape: {}".format(input_matrix.shape))
    signed_mat = diag*input_matrix
    # print(signed_mat.shape)
    S = fastwht(signed_mat)*shift_bit_length(nrows) # shift bit length is normalising factor
    sample = np.random.choice(nrows, sketch_size, replace=False)
    #sample.sort()
    # number from num_rows_data universe
    S = S[sample]
    S = (sketch_size)**(-0.5)*S
    return S


#
# def hadamard_transform(data):
#     '''
#     Real Fast Fourier Transform (FFT) Independently Applied to Each Column of A
#
#     Input
#         a_mat: n-by-d dense np matrix.
#
#     Output
#         c_mat: n-by-d matrix C = F * A.
#         Here F is the n-by-n orthogonal real FFT matrix (not explicitly formed)
#
#     Notice that $C^T * C = A^T * A$;
#     however, $C * C^T = A * A^T$ is not true.
#     '''
#     n = data.shape[0]
#     H_mat = np.fft.fft(data, n=None, axis=0) / np.sqrt(n)
#     if n % 2 == 1:
#         cutoff_int = int((n+1) / 2)
#         idx_real_vec = list(range(1, cutoff_int))
#         idx_imag_vec = list(range(cutoff_int, n))
#     else:
#         cutoff_int = int(n/2)
#         idx_real_vec = list(range(1, cutoff_int))
#         idx_imag_vec = list(range(cutoff_int+1, n))
#     sketch = H_mat.real
#     sketch[idx_real_vec, :] *= np.sqrt(2)
#     sketch[idx_imag_vec, :] = H_mat[idx_imag_vec, :].imag * np.sqrt(2)
#     return sketch

#
# def srht_transform(data, sketch_size):
#     '''
#     Subsampled Randomized Fourier Transform (SRFT) for Dense Matrix
#
#     Input
#         a_mat: m-by-n dense np matrix;
#         s_int: sketch size.
#
#     Output
#         c_mat: m-by-s sketch C = A * S.
#         Here S is the sketching matrix (not explicitly formed)
#     '''
#     n,d = data.shape
#     random_signs = np.random.choice(2, n) * 2 - 1
#     sample_ids = np.random.choice(n, sketch_size, replace=False)
#     data = random_signs.reshape(n,1) * data
#     data = hadamard_transform(data)
#     summary = data[sample_ids, :] * np.sqrt(n / sketch_size)
#     return summary

# @jit
# def realfft_col(a_mat):
#     '''
#     Real Fast Fourier Transform (FFT) Independently Applied to Each Column of A
#
#     Input
#         a_mat: n-by-d dense NumPy matrix.
#
#     Output
#         c_mat: n-by-d matrix C = F * A.
#         Here F is the n-by-n orthogonal real FFT matrix (not explicitly formed)
#
#     Notice that $C^T * C = A^T * A$;
#     however, $C * C^T = A * A^T$ is not true.
#     '''
#     n_int = a_mat.shape[0]
#     fft_mat = np.fft.fft(a_mat, n=None, axis=0) / np.sqrt(n_int)
#     if n_int % 2 == 1:
#         cutoff_int = int((n_int+1) / 2)
#         idx_real_vec = list(range(1, cutoff_int))
#         idx_imag_vec = list(range(cutoff_int, n_int))
#     else:
#         cutoff_int = int(n_int/2)
#         idx_real_vec = list(range(1, cutoff_int))
#         idx_imag_vec = list(range(cutoff_int+1, n_int))
#     c_mat = fft_mat.real
#     c_mat[idx_real_vec, :] *= np.sqrt(2)
#     c_mat[idx_imag_vec, :] = fft_mat[idx_imag_vec, :].imag * np.sqrt(2)
#     return c_mat
#
# @jit
# def realfft_row(a_mat):
#     '''
#     Real Fast Fourier Transform (FFT) Independently Applied to Each Row of A
#
#     Input
#         a_mat: m-by-n dense NumPy matrix.
#
#     Output
#         c_mat: m-by-n matrix C = A * F.
#         Here F is the n-by-n orthogonal real FFT matrix (not explicitly formed)
#
#     Notice that $C * C^T = A * A^T$;
#     however, $C^T * C = A^T * A$ is not true.
#     '''
#     n_int = a_mat.shape[1]
#     fft_mat = np.fft.fft(a_mat, n=None, axis=1) / np.sqrt(n_int)
#     if n_int % 2 == 1:
#         cutoff_int = int((n_int+1) / 2)
#         idx_real_vec = list(range(1, cutoff_int))
#         idx_imag_vec = list(range(cutoff_int, n_int))
#     else:
#         cutoff_int = int(n_int/2)
#         idx_real_vec = list(range(1, cutoff_int))
#         idx_imag_vec = list(range(cutoff_int+1, n_int))
#     c_mat = fft_mat.real
#     c_mat[:, idx_real_vec] *= np.sqrt(2)
#     c_mat[:, idx_imag_vec] = fft_mat[:, idx_imag_vec].imag * np.sqrt(2)
#     return c_mat
#
# @jit
# def fast_srft(a_mat, s_int):
#     '''
#     Subsampled Randomized Fourier Transform (SRFT) for Dense Matrix
#
#     Input
#         a_mat: m-by-n dense NumPy matrix;
#         s_int: sketch size.
#
#     Output
#         c_mat: m-by-s sketch C = A * S.
#         Here S is the sketching matrix (not explicitly formed)
#     '''
#     n_int = a_mat.shape[1]
#     sign_vec = np.random.choice(2, n_int) * 2 - 1
#     idx_vec = np.random.choice(n_int, s_int, replace=False)
#     a_mat = a_mat * sign_vec.reshape(1, n_int)
#     a_mat = realfft_row(a_mat)
#     c_mat = a_mat[:, idx_vec] * np.sqrt(n_int / s_int)
#     return c_mat
#
# @jit
# def srft2(a_mat, b_mat, s_int):
#     '''
#     Subsampled Randomized Fourier Transform (SRFT) for Dense Matrix
#
#     Input
#         a_mat: m-by-n dense NumPy matrix;
#         b_mat: d-by-n dense NumPy matrix;
#         s_int: sketch size.
#
#     Output
#         c_mat: m-by-s sketch C = A * S;
#         d_mat: d-by-s sketch D = B * S.
#         Here S is the sketching matrix (not explicitly formed)
#     '''
#     n_int = a_mat.shape[1]
#     sign_vec = np.random.choice(2, n_int) * 2 - 1
#     idx_vec = np.random.choice(n_int, s_int, replace=False)
#
#     a_mat = a_mat * sign_vec.reshape(1, n_int)
#     a_mat = realfft_row(a_mat)
#     c_mat = a_mat[:, idx_vec] * np.sqrt(n_int / s_int)
#
#     b_mat = b_mat * sign_vec.reshape(1, n_int)
#     b_mat = realfft_row(b_mat)
#     d_mat = b_mat[:, idx_vec] * np.sqrt(n_int / s_int)
#     return c_mat, d_mat
#



class SRHT(Sketch):
    '''Performs the SRHT to compute the sketch

    Parameters
    ----------
    sketch_dimension: int - number of rows to compress into.

    References
    -----------
    https://arxiv.org/abs/1411.4357 -- DP Woodruff
    '''

    def __init__(self, data, sketch_dimension, random_state=None, second_data=None, timing=False):
        if random_state is not None or second_data is not None:
            super(SRHT,self).__init__(data, sketch_dimension,\
                                                        random_state, second_data)
        else:
            super(SRHT,self).__init__(data, sketch_dimension)
        self.timing = timing

        if sp.sparse.issparse(data):
            raise TypeError("Expected a dense input matrix, got scipy sparse")

    def sketch(self, data):
        #S = srht_transform(data, self.sketch_dimension)
        if self.timing:
            start = default_timer()
            S = srht_transform1(data, self.sketch_dimension)
            #S = (fast_srft(data.T, self.sketch_dimension)).T
            end = default_timer() - start
            return S, end
        else:
            S = srht_transform1(data, self.sketch_dimension)
            #S = (fast_srft(data.T, self.sketch_dimension)).T
            return S

    # def sketch_product(self, first_data, second_data):
    #     '''
    #     sketches self and another dataset.
    #     Generates a sketch of the identity to act as the same sketching matrix
    #     on both datasets to pass the test.
    #     '''
    #     S = self.sketch(I)
    #     S_X, S_Y = srft2(first_data, second_data, self.sketch_dimension)
    #
    #     return S_X, S_Y
