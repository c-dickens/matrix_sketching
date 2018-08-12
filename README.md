# Matrix Sketching

The aim of the project is to be a fully reproducible and usable library for various matrix sketching tasks.
The initial problem is to investigate how different random transforms can be used as preconditioners for Convex Constrained Least Squares problems (CCLSQ), particularly with a view to exploiting sparse embeddings for fast-to-compute preconditioners.

## Installation:
1. `git clone` the repo
2. `cd matrix_sketching`
3. `pip install -r requirements.txt`
4. `cd matrix_sketching/lib`
5. `git clone https://bitbucket.org/vegarant/fastwht.git` --> then run install
in here by `cd python`, `python setup.py`, `python test.py`
6. Get the directory path for `fastwht/python` which should be `your_path_name =
*/matrix_sketching/lib/fastwht/python`
7. Find `.bash_profile` or equivalent and add `export PYTHONPATH=$PYTHONPATH:your_path_name`,
at the final line of the `bash_profile`, finally save then `source .bash_profile`.
8. Open ipython, do `import sys --> sys.path` and check that `your_path_name`
is displayed.
9. Go back to the `matrix_sketching` directory and run the tests.


## Roadmap:
1. ~~Get IHS LASSO solver working.~~ [26/07/2018] DONE - now incorporate into class based method.
2. Get IHS SVM solver working.
3. Add timing functionality to the sketching objects.
4. Include `sparseJLT` methods.

### Notes
-  Very few iterations are needed to achieve error < `10^-10` for the IHS lasso
- The scaling for the constrained QP is very severe as `d` increases (no surprises given that QP are `d^3` in the worst case) and in this increased `d` domain it may not be favourable to use the sketching technique.
- However, if instead one can accept a heuristic solution then the penalised form can be used.
- How does varying the sketch size affect the convergence of the problem?  So far it seems that very small sketches work fine.
- At what point, if any, do we need to switch from the fast sparse Count Sketch
implementation to the dense one?
- I tested against the PyRLA implementation (hence the extra commented code
  in the class definition) of the SRHT but mine seemed to be
faster so used that one throughout.
- using the `_countsketch_fast` function is faster than the dense function
despite having to iterate over more values.

## Experiments
1. `verify_ihs_paper.py` -- script to show how Count Sketch fits into the IHS
sketching scheme.
  [x] - Unconstrained regression  as `n` grows and `d` and sketch dim are fixed.
  _PLOTTING Done_
  [x] - Unconstrained regression as the number of iterations grows, `n,d` and
  sketch size fixed but the sketches and the sketching dimension are varied.
  _Plotting Done_
  [x] - Unconstrained regression with all sketches. Vary `d` and fix a corresponding `n`. Sketch size and num iters
  are fixed.  _plotting done_.
  [] - Tidy up axes and labels etc.

2. `sketching_lasso_synthetic.py` -- the sketched lasso problem on various synthetic
datasets.

## Datasets
- `YearPredictionMSD` - taken from UCI ML repo.  Download and usage from
`get_data.py` and shell script in data repo.
- `rail2586` - Taken from Florida Sparse Matrix Repo.  Download MATLAB file and
save `X = X = (Problem.A)'  `.  Convert `.mat` to a `.txt` so others can
reproduce the results.
_to try_
- `landmark` - from Florida collection
- `specular` - as above
- `abtaha2` - as above
- `Rucci1` - as above (take sample of columns)
### Credits
I have used code from https://bitbucket.org/vegarant/fastwht to construct the SRHT and used https://github.com/wangshusen/PyRLA as inspiration, although more features and test suites have been added.
