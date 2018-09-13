'''
Script to get real dataset metadata i.e size etc.
'''
from pprint import PrettyPrinter
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

    with open('data_metadata.json') as f:
        already_seen_data = json.load(f)
    print(already_seen_data.keys())
    print(metadata)

    for data in datasets:
        print("-"*80)
        print("Dataset: {}".format(data))
        input_file = datasets[data]["filepath"]
        sparse_flag = False # a check to say if sparse data is found.
        # Defaults to False unles the sparse matrix can be read in.

        if data in already_seen_data.keys():
            # already computed data summary for this set.
            print("Already summarised so moving on.")
            continue
        else:
            print("Summarising new dataset {}".format(data))

            dense_data = np.load(input_file)
            n,d = dense_data.shape
            if (n,d) == (1,2):
                print("SAVED IN SPARSE FORMAT SO NEED SOME EXTRA WORK.")
                items_in_list = [item for sublist in dense_data for item in sublist]
                #print(items_in_list[0])
                sparse_data = items_in_list[0]
                n,d = sparse_data.shape
                print(n,d)
                aspect_ratio = d/n
                nnz = sparse_data.getnnz()
                density = nnz / (n*d)
                # nb in here we are really using a low rank approx for the
                # leverage scores as we are projecting onto NUM_SING_VECTORS
                # Can grow up to d but takes longer to compute the scores.
                NUM_VECTORS_FOR_PROJECTION = np.int(d)-1
                U, _,_ = sparse.linalg.svds(sparse_data,NUM_VECTORS_FOR_PROJECTION)
                lev_scores = np.linalg.norm(U, axis=1)**2
                coherence = np.max(lev_scores)
                coherence_ratio = coherence / np.min(lev_scores)
                rank = d


            else:

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
    pretty = PrettyPrinter(indent=4)
    pretty.pprint(metadata)

    new_dict = {}
    already_seen_data.update(metadata)
    pretty.pprint(already_seen_data)
    with open('data_metadata.json', 'w') as outfile:
       json.dump(already_seen_data, outfile)


if __name__ == "__main__":
    main()
