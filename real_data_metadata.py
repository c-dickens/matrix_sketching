'''
Script to get real dataset metadata i.e size etc.
'''
import json
import itertools
import numpy as np
import scipy as sp
from sklearn.preprocessing import StandardScaler
from scipy import sparse
from scipy.sparse import load_npz, coo_matrix
import datasets_config


def main():
    metadata = {}
    datasets = datasets_config.datasets

    for data in datasets:
        if data != "kdd":
            metadata[data] = {}

    print(metadata)



    for data in datasets:
        print("-"*80)
        print("Dataset: {}".format(data))
        input_file = datasets[data]["filepath"]
        sparse_flag = False # a check to say if sparse data is found.
        # Defaults to False unles the sparse matrix can be read in.

        if data is "kdd":
            print("Ignoring this one.")
            continue

        dense_data = np.load(input_file)
        n,d = dense_data.shape
        if (n,d) == (1,2):
            # this checks if the saved file is csr matrix and if so
            # continues the loop as the code needs to be different.
            print("SAVED IN SPARSE FORMAT SO SKIPPING FOR NOW.")
            continue
        aspect_ratio = d/n
        nnz = np.count_nonzero(dense_data)
        density = nnz/(n*d)
        q,_ = np.linalg.qr(dense_data)
        lev_scores = np.linalg.norm(q, axis=1)**2
        coherence = np.max(lev_scores)
        coherence_ratio = coherence / np.min(lev_scores)
        rank = np.linalg.matrix_rank(dense_data)


        print("Shape: {}".format((n,d)))
        print("Aspect ratio : {}".format(aspect_ratio))
        print("Density: {} ".format(density))
        print("Lev score sum: {}".format(np.sum(lev_scores)))
        print("Coherence {}".format(coherence))
        print("Coherence ratio: {} ".format(coherence / np.min(lev_scores[lev_scores > 0])))

        metadata[data] = {
                    "shape"           : (n,d),
                    "aspect ratio"    : aspect_ratio,
                    "density"         : density,
                    "rank"            : np.int(round(np.sum(lev_scores))),
                    "coherence"       : coherence,
                    "coherence_ratio" : coherence / np.min(lev_scores[lev_scores > 0])
        }

    with open('data_metadata.json', 'w') as outfile:
       json.dump(metadata, outfile)


if __name__ == "__main__":
    main()
