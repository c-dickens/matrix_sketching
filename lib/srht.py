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
    signed_mat = diag*input_matrix
    S = fastwht(signed_mat)*shift_bit_length(nrows) # shift bit length is normalising factor
    sample = np.random.choice(nrows, sketch_size, replace=False)
    #sample.sort()
    # number from num_rows_data universe
    S = S[sample]
    S = (sketch_size)**(-0.5)*S
    return S



def hadamard_transform(data):
    '''
    Real Fast Fourier Transform (FFT) Independently Applied to Each Column of A

    Input
        a_mat: n-by-d dense np matrix.

    Output
        c_mat: n-by-d matrix C = F * A.
        Here F is the n-by-n orthogonal real FFT matrix (not explicitly formed)

    Notice that $C^T * C = A^T * A$;
    however, $C * C^T = A * A^T$ is not true.
    '''
    n = data.shape[0]
    H_mat = np.fft.fft(data, n=None, axis=0) / np.sqrt(n)
    if n % 2 == 1:
        cutoff_int = int((n+1) / 2)
        idx_real_vec = list(range(1, cutoff_int))
        idx_imag_vec = list(range(cutoff_int, n))
    else:
        cutoff_int = int(n/2)
        idx_real_vec = list(range(1, cutoff_int))
        idx_imag_vec = list(range(cutoff_int+1, n))
    sketch = H_mat.real
    sketch[idx_real_vec, :] *= np.sqrt(2)
    sketch[idx_imag_vec, :] = H_mat[idx_imag_vec, :].imag * np.sqrt(2)
    return sketch


def srht_transform(data, sketch_size):
    '''
    Subsampled Randomized Fourier Transform (SRFT) for Dense Matrix

    Input
        a_mat: m-by-n dense np matrix;
        s_int: sketch size.

    Output
        c_mat: m-by-s sketch C = A * S.
        Here S is the sketching matrix (not explicitly formed)
    '''
    n,d = data.shape
    random_signs = np.random.choice(2, n) * 2 - 1
    sample_ids = np.random.choice(n, sketch_size, replace=False)
    data = random_signs.reshape(n,1) * data
    data = hadamard_transform(data)
    summary = data[sample_ids, :] * np.sqrt(n / sketch_size)
    return summary




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

    def sketch(self, data):
        #S = srht_transform(data, self.sketch_dimension)
        if self.timing:
            start = default_timer()
            S = srht_transform1(data, self.sketch_dimension)
            end = default_timer() - start
            return S, end
        else:
            S = srht_transform1(data, self.sketch_dimension)
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
