Reference: [1]: "Projective Splitting with Forward Steps only Requires Continuity", Patrick R. Johnstone, Jonathan Eckstein, https://arxiv.org/pdf/1809.07180.pdf

Optimization problem (robust fused lasso):

min (1/p)*||Ax - b||_p^p + lam1*||Dx||_1 + lam2*||x||_1

To run the experiments in [1]:
1. save both robustFusedLasso.py and runRobustFusedLasso.py in the same directory
2. Run python with the input file runRobustFusedLasso.py from that directory. 
 - Eg. in Linux simply enter python runRobustFusedLasso.py 
3. To select between the three experiments in the paper, set the variable Exp to 1,2, or 3. 

There are two python files.

runFusedLasso.py runs the experiments in [1] using the functions defined in robustFusedLasso.py 

robustFusedLasso.py defines the following functions:

1. ps(): projective splitting applied to the robused fused lasso problem. 
2. backTrack(): internal function of ps() which takes care of backtracking.
3. subg(): the subgradient method applied to the robust fused lasso problem.
4. proxSubG(): the proximal subgradient method applied to the robust fused lasso problem.
5. tseng_pd(): Tseng's method applied to solving the robust fused lasso problem via the monotone+skew primal-dual inclusion.
6. theBigProx(): Internal function used in tseng_pd(), takes care of computing the resolvent (prox) in the primal-dual space. 
7. diffOperator(): Difference operator, matrix D, defined in the robust fused lasso problem.
8. adjointD(): adjoint (transpose) of D.
9. proxL1(): prox w.r.t. ell_1 norm, aka soft thresholding operator
10. projLInf(): Projection onto the ell_infinity ball




