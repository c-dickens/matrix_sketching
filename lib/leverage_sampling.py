import numpy as np
from . import Sketch


def exact_leverage_score(data):
    '''Computes the exact levergae scores of matrix 'data'.

    Input:
        data - ndarray matrix of size n x d

    Output:
        leverage_scores - one dimensional array containing the
                          exact leverage scores.
    '''
    n,d = data.shape
    u,_,_ = np.linalg.svd(data, full_matrices=False) # Try this with QR?
    leverage_scores = np.sum(u**2,axis=1) # axis = 1 to get row leverage
    return leverage_scores

def row_sampling(data, sketch_size, leverage_scores):
    '''
    Samples the rows of a matrix according to a distribution defined by
    leverage_scores.

    Input:
        data - ndarray
        sketch_size - int, the number rows to sample.
        leverage_scores - the leverage_scores (or approximations) to define
                          the sampling distribution.
    '''

    n,d = data.shape
    sample_probabilities = leverage_scores/np.sum(leverage_scores)
    print("Sample probability length: {}".format(sample_probabilities.shape))
    sampled_ids = np.random.choice(n,sketch_size,replace=False, p=sample_probabilities)
    rescaling_vector = np.sqrt(sketch_size*sample_probabilities[sampled_ids]) + 1e-10
    # Not sure about addition term, maybe to fix edge case if a zero is sampled?
    S = data[sampled_ids,:] / rescaling_vector.reshape(len(rescaling_vector),1)
    return S, sampled_ids

class LeverageScoreSampler(Sketch):
    '''
    Class instantiation of leverage score sampling.

    References:
    [1] - https://arxiv.org/abs/1104.5557
    '''
    def __init__(self, data, sketch_dimension, random_state=None, second_data=None):
        if random_state is not None or second_data is not None:
            super(LeverageScoreSampler,self).__init__(data, sketch_dimension,\
                                                        random_state, second_data)
        else:
            super(LeverageScoreSampler,self).__init__(data, sketch_dimension)

    def get_leverage_scores(self):
        '''Computes the leverage scores of the data matrix.
        To do: Add a method argument.'''
        leverage_scores = exact_leverage_score(self.data)
        return leverage_scores


    def sketch(self,data,get_sampled_ids=False):
        '''
        method : Exact --  calls
                 approx --
                 approx_fast --
        Exact uses the exact leverage score algorithm and approx uses the fast
        approximation.
        '''
        scores = exact_leverage_score(data)
        summary, sampled_ids = row_sampling(data, self.sketch_dimension, scores)

        if get_sampled_ids:
            return summary, sampled_ids
        else:
            return summary
