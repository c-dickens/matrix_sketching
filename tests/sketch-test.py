'''
Code to do a basic test on the sketching techniques to check that they run
'''

def main():
    import numpy as np
    import matrix_sketching.gaussian as rp


    random_state = 10
    np.random.seed(random_state)

    matrix_rows = 10000
    matrix_columns = 100
    sketch_dimension = 10*matrix_columns

    matrix = np.random.randn(matrix_rows, matrix_columns)
    true_norm = np.linalg.norm(matrix,ord='fro')**2

    summary = rp.GaussianSketch(data=matrix,
                             sketch_dimension=sketch_dimension,
                             random_state=random_state)


    S_A = summary.sketch()
    estimate_norm = np.linalg.norm(S_A,ord='fro')**2
    print("Norm ratio: {}".format(true_norm/estimate_norm))



if __name__ == "__main__":
    import sys

    root_dir = '..'
    sys.path.append(root_dir)
    main()
