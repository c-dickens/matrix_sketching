# Matrix Sketching

The aim of the project is to be a fully reproducible and usable library for various matrix sketching tasks.
The initial problem is to investigate how different random transforms can be used as preconditioners for Convex Constrained Least Squares problems (CCLSQ), particularly with a view to exploiting sparse embeddings for fast-to-compute preconditioners.


## Roadmap:
1. ~~Get IHS LASSO solver working.~~ [26/07/2018] DONE - now incorporate into class based method.
2. Get IHS SVM solver working.

### Credits
I have used code from https://bitbucket.org/vegarant/fastwht to construct the SRHT and used https://github.com/wangshusen/PyRLA as inspiration, although more features and test suites have been added.