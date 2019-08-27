'''
#references:
#[1]: "Projective Splitting with Forward Steps only Requires Continuity", Patrick R. Johnstone, Jonathan Eckstein, https://arxiv.org/pdf/1809.07180.pdf

# Functions defined in this file:
1. ps() - projective splitting applied to the robust fused lasso problem
2.
'''
import numpy as np
from matplotlib import pyplot as plt
import time



def ps(d,iter,rho1,rho2,rho3,theProx,theGrad,theFunc,gamma,lam1,lam2,Delta):
    '''
    Paper reference:
    [1] "Projective Splitting with Forward Steps only Requires Continuity", Patrick R. Johnstone, Jonathan Eckstein, https://arxiv.org/pdf/1809.07180.pdf
    projective splitting applied to the problem:
    min F(x) = f(x)+lam1*np.linalg.norm(D.dot(x),1)+lam2*np.linalg.norm(x,1)
    f(x) is accessed by its gradient theGrad(), which is method passed in as an argument
    the prox of the ell_1 norm is passed in as theProx() (defined in this file as function proxL1)
    The objective function F(x) can be evaluated with theFunc
    theProx, theGrad, theFunc are user-defined functions
    For the fused robust lasso, we have defined them in the file runRobustFusedLasso.py
    Other inputs...
        d: dimension of the problem
        iter: number of iterations
        rho1,rho2,rho3 stepsizes
        rho3 is discovered by backtracking so this input is just the initial stepsize
        gamma: as in [1], parameter of projective splitting
        lam1,lam2: see definition of the robust fused lasso problem
        Delta: parameter for the backtracking procedure, see [1]
    Outputs
       fx2: list of function values f(x_2^k) at each iteration
       x2: final value of x_2^k
       total_mults: list of multiplications by Ax for each iteration
       times: list of times for each iteration
    '''

    x2 = np.zeros(d)
    z = np.zeros(d)
    w1 = np.zeros(d-1)
    w2 = np.zeros(d)
    fx2 = [theFunc(x2)]
    k = 0
    total_mults = [0]

    notDone = True
    times = [0]
    tstart = time.time()
    while notDone:

        Dz = diffOperator(z)
        t1 =  Dz + rho1 * w1
        x1 = theProx(t1,rho1*lam1)
        y1 = (1 / rho1) * (t1 - x1)

        Dtw1 = adjointD(w1)
        w3 = - Dtw1 - w2

        t2 = z + rho2 * w2
        x2 = theProx(t2, rho2 * lam2)
        y2 = (1 / rho2) * (t2 - x2)

        [x3, y3, rho3, mults] = backTrack(z, rho3, w3, theGrad, Delta)
        total_mults.append(total_mults[k] + mults)


        phi = (Dz - x1).T.dot(y1 - w1) + (z - x2).T.dot(y2 - w2) + (z - x3).T.dot(y3 - w3)
        Dty1 = adjointD(y1)
        gradz = Dty1 + y2 + y3
        Dx3 = diffOperator(x3)
        gradw1 = x1 - Dx3
        gradw2 = x2 - x3
        normGradsq = gamma ** (-1) * np.linalg.norm(gradz) ** 2 + np.linalg.norm(gradw1) ** 2 + np.linalg.norm(gradw2) ** 2

        if (normGradsq > 0):
            z = z - gamma ** (-1) * (phi / normGradsq) * gradz
            w1 = w1 - (phi / normGradsq) * gradw1
            w2 = w2 - (phi / normGradsq) * gradw2
        else:
            print("gradient of hyperplane is 0")
            notDone = False


        fx2.append(theFunc(x2))
        tstamp = time.time()
        times.append(tstamp - tstart)
        k+=1
        if(k>=iter):
            notDone = False

    return [fx2,x2,total_mults,times]

def backTrack(z,rho,w,theGrad,Delta):
    # backtracking subroutine for ps()
    keepBTing = True
    Bz = theGrad(z)
    mults=2

    while(keepBTing):
        xnew = z - rho * (Bz - w)
        ynew = theGrad(xnew)
        mults+=2

        if(Delta*np.linalg.norm(z-xnew)**2 - (z - xnew).dot(ynew - w) <=0):
            #backtracking complete
            keepBTing = False
        else:
            rho = 0.7*rho

    return [xnew,ynew,rho,mults]


def subg(d,iter,theGrad,theSubGrad,lam1,lam2,alpha_0,r,theFunc):
    '''
    the subgradient method applied to the robust fused lasso problem
    min F(x) = f(x)+lam1*np.linalg.norm(D.dot(x),1)+lam2*np.linalg.norm(x,1)
    inputs:
        d: variable dimension
        iter: number of iteratrions
        theGrad: procedure for computing gradient of the differentiable plotResids
        theSubGrad: procedure for computing subgradients w.r.t. nonsmooth functions, i.e. ell_1 norm
        lam1,lam2: see definition of the robust fused lasso problem
        alpha_0,r: the subgradient uses stepsizes alpha_k = alpha_0 (k+1)^{-r}
        theFunc: procedure to evaluate F(x)
    outputs:
        x: the final Variable
        Fx: list of function values at each iteration
        times: running times at each iteration
    Note that we do not need to record the number of matrix multiplies because the subgradient
    method does not use backtracking and therefore always uses exactly to matrix multiplies per iteration
    to compute g
    '''
    x = np.zeros(d)
    Fx = [theFunc(x)]
    times = [0]
    tstart = time.time()
    for k in range(iter):
        alpha = alpha_0*(k+1)**(-r)
        g = theGrad(x)+lam1*adjointD(theSubGrad(diffOperator(x))) + lam2*theSubGrad(x)
        x = x - alpha*g
        Fx.append(theFunc(x))
        tstamp = time.time()
        times.append(tstamp - tstart)
    return [x,Fx,times]


def proxSubG(d,iter,theGrad,theProx,alpha_0,r,lam2,theFunc,lam1,theSubGrad):
    '''
    the proximal subgradient method applied to the robust fused lasso problem
    min F(x) = f(x)+lam1*np.linalg.norm(D.dot(x),1)+lam2*np.linalg.norm(x,1)
    inputs:
        d: variable dimension
        iter: number of iteratrions
        theGrad: procedure for computing gradient of the differentiable plotResids
        theSubGrad: procedure for computing subgradients w.r.t. nonsmooth functions, i.e. ell_1 norm
        theProx: procedure to compute the prox w.r.t. ell_1 norm
        lam1,lam2: see definition of the robust fused lasso problem
        alpha_0,r: the subgradient uses stepsizes alpha_k = alpha_0 (k+1)^{-r}
        theFunc: procedure to evaluate F(x)
    outputs:
        x: the final Variable
        Fx: list of function values at each iteration
        times: running times at each iteration
    Note that we do not need to record the number of matrix multiplies because the subgradient
    method does not use backtracking and therefore always uses exactly to matrix multiplies per iteration
    to compute g
    '''
    x = np.zeros(d)
    Fx = [theFunc(x)]
    times = [0]
    tstart = time.time()
    for k in range(iter):
        alpha = alpha_0*(k+1)**(-r)
        g = theGrad(x)+lam1*adjointD(theSubGrad(diffOperator(x)))
        x = theProx(x-alpha*g,alpha*lam2)
        Fx.append(theFunc(x))
        tstamp = time.time()
        times.append(tstamp-tstart)
    return [x,Fx,times]


def tseng_pd(d,iter,theGrad,alpha,theProxL1star,theta,theFunc,lam1,lam2,gamma):
    '''
    Tseng's method applied to solving the robust fused lasso problem via the monotone+skew primal-dual inclusion
    inputs:
        d: variable dimension
        iter: number of iteratrions
        theGrad: procedure for computing gradient of the differentiable plotResids
        theProx: procedure to compute the prox w.r.t. ell_1 norm
        lam1,lam2: see definition of the robust fused lasso problem
        theFunc: procedure to evaluate F(x)
        alpha: stepsize
        gamma: used in the preconditioner (e.g. variable metric)
        theProxL1star: procedure to compute the prox w.r.t. the conjugate function of the ell_1 norm
    Outputs:
        x: the final Variable
        Fx: list of function values at each iteration
        total_mults: number of matrix multiplies by A per iteration
        times: running times at each iteration
    '''
    x = np.zeros(d)
    w1 = np.zeros(d)
    w2 = np.zeros(d-1)
    Fx = [theFunc(x)]
    total_mults = [0]
    times = [0]
    tstart = time.time()

    for k in range(iter):
        #compute Ap
        Ap1 = -x
        Ap2 = -diffOperator(x)
        Ap3 = w1 + adjointD(w2) + theGrad(x)
        mults = 2
        keepBT = True
        alpha = alpha
        while keepBT:

            pbar = theBigProx(w1 - gamma*alpha*Ap1,w2 - gamma*alpha*Ap2,x -alpha*Ap3,theProxL1star,lam1,lam2)
            Apbar1 = -pbar[2]
            Apbar2 = -diffOperator(pbar[2])
            Apbar3 = pbar[0] + adjointD(pbar[1]) + theGrad(pbar[2])
            mults+=2
            totalNorm \
                = np.sqrt(gamma*np.linalg.norm(Apbar1-Ap1)**2+gamma*np.linalg.norm(Apbar2-Ap2)**2+np.linalg.norm(Apbar3-Ap3)**2)
            totalNorm2 \
                = np.sqrt(gamma**(-1)*np.linalg.norm(pbar[0]-w1)**2+gamma**(-1)*np.linalg.norm(pbar[1]-w2)**2+np.linalg.norm(pbar[2]-x)**2)

            if(alpha*totalNorm<=theta*totalNorm2):
                keepBT = False
            else:
                alpha = 0.7*alpha

        w1 = pbar[0] - gamma*alpha * (Apbar1 - Ap1)
        w2 = pbar[1] - gamma*alpha * (Apbar2 - Ap2)
        x  = pbar[2] - alpha * (Apbar3 - Ap3)
        Fx.append(theFunc(x))
        tstamp = time.time()
        times.append(tstamp-tstart)
        total_mults.append(total_mults[k]+mults)
    return [x,Fx,total_mults,times]

def theBigProx(a,b,c,theProxL1Star,lam1,lam2):
    '''
    subroutine used in tseng_pd
    Computes the resolvent (i.e. prox) in the monotone+skew primal-dual product space
    '''
    out1 = theProxL1Star(a,lam2)
    out2 = theProxL1Star(b,lam1)
    out3 = c
    return [out1,out2,out3]

def diffOperator(x):
    # finite differences operator used in the robust fused lasso problem
    d = len(x)
    return x[1:d] - x[0:(d-1)]

def adjointD(y):
    # adjoint of the finite differences operator
    p = len(y)
    #reverse diffs with 0 padding
    middlePart = y[0:(p-1)]-y[1:p]
    return np.concatenate([ np.array([-y[0]]),middlePart , np.array([y[p-1]]) ])

def proxL1(a,rholam):
    # prox w.r.t. ell_1 norm, aka soft thresholding operator
    x = (a> rholam)*(a-rholam)
    x+= (a<-rholam)*(a+rholam)
    return x

def projLInf(x,lam):
    # projection onto the ell_infinity ball
    # this is the prox of the conjugate function of the ell_1 norm and is used in Tseng-pd
    return (x>=-lam)*(x<=lam)*x + (x>lam)*lam - (x<-lam)*lam
