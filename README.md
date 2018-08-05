# Matrix Sketching

The aim of the project is to be a fully reproducible and usable library for various matrix sketching tasks.
The initial problem is to investigate how different random transforms can be used as preconditioners for Convex Constrained Least Squares problems (CCLSQ), particularly with a view to exploiting sparse embeddings for fast-to-compute preconditioners.


## Roadmap:
1. ~~Get IHS LASSO solver working.~~ [26/07/2018] DONE - now incorporate into class based method.
2. Get IHS SVM solver working.
3. Add timing functionality to the sketching objects.

### Notes
-  Very few iterations are needed to achieve error < `10^-10` for the IHS lasso
- The scaling for the constrained QP is very severe as `d` increases (no surprises given that QP are `d^3` in the worst case) and in this increased `d` domain it may not be favourable to use the sketching technique.
- However, if instead one can accept a heuristic solution then the penalised form can be used.
- How does varying the sketch size affect the convergence of the problem?  So far it seems that very small sketches work fine.
- At what point, if any, do we need to switch from the fast sparse Count Sketch implementation to the dense one?

## Experiments
1. `verify_ihs_paper.py` -- script to show how Count Sketch fits into the IHS
sketching scheme.
  [x] - Unconstrained regression  as `n` grows and `d` and sketch dim are fixed.
  _PLOTTING INCOMPLETE_
  [x] - Unconstrained regression as the number of iterations grows, `n,d` and
  sketch size fixed but the sketches and the sketching dimension are varied.
  _Plotting Done_
  [x] - Unconstrained regression with all sketches. Vary `d` and fix a corresponding `n`. Sketch size and num iters are fixed.  _plotting almost done_.
2. `sketching_lasso_synthetic.py` -- the sketched lasso problem on various synthetic
datasets.

### Credits
I have used code from https://bitbucket.org/vegarant/fastwht to construct the SRHT and used https://github.com/wangshusen/PyRLA as inspiration, although more features and test suites have been added.
