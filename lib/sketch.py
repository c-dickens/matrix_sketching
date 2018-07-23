import numpy as np
import numba
from numba import jit


class Sketch(object):
    def __init__(self, data, sketch_dimension, random_state=None, second_data=None):
        #pass
        self.data = data
        self.sketch_dimension = sketch_dimension
        self.num_rows, self.num_cols = data.shape
        self.random_state = random_state
        self.second_data = second_data
