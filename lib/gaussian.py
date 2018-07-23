'''
Gaussian sketching matrices
'''

import numpy as np
from . import Sketch

class GaussianSketch(Sketch):
    '''Reduces dimensionality by using a Gaussian random projection.

    Generates a summary B = S*A where S is a matrix whose entries are drawn
    from N(0,1/sqrt(k)) and k is the number of rows of S.

    Parameters
    ----------
    sketch_dimension: int - number of rows to compress into.
    '''

    def __init__(self, data, sketch_dimension, random_state=None, second_data=None):
        if random_state is not None or second_data is not None:
            super(GaussianSketch,self).__init__(data, sketch_dimension,\
                                                        random_state, second_data)
        else:
            super(GaussianSketch,self).__init__(data, sketch_dimension)


    def sketch(self,data):
        S = (self.sketch_dimension)**(-0.5)*np.random.randn(\
                                            self.sketch_dimension, self.num_rows)
        return np.dot(S, data)

    def sketch_product(self, first_data, second_data):
        '''
        sketches self and another dataset.
        Generates a sketch of the identity to act as the same sketching matrix
        on both datasets to pass the test.
        '''

        I = np.identity(first_data.shape[0])
        S = self.sketch(I)
        S_X = S@first_data
        S_Y = S@second_data

        return S_X, S_Y
