'''
This file runs the experiments in
[1]: "Projective Splitting with Forward Steps only Requires Continuity", Patrick R. Johnstone, Jonathan Eckstein, https://arxiv.org/pdf/1809.07180.pdf
There are three experiments in the paper.
To choose each experiment, set the variable Exp as follows:
Exp = 1 for p=1.7
Exp = 2 for p=1.3
Exp = 3 for p=1.5
The following algorithms are run:
projective splitting (ps)
the subgradient method (subg)
the proximal subgradient method (prox-sg)
Tseng's method applied to the primal-dual inclusion (Tseng-pd)
These algorithms are defined in the file robustFusedLasso.py
'''

import numpy as np
import robustFusedLasso as fl
import time
from matplotlib import pyplot as plt

d = 2000  #dimension of the problem
n = 1000 # number of measurements, i.e. rows of A

Exp = 1 # select which experiment Exp=1,2, or 3.


if Exp==1:
    #Experiment 1
    p = 1.7
    lam1 = 1.0
    lam2 = 1.0
elif Exp==2:
    #Experiment 2
    p = 1.3
    lam1 = 1.0
    lam2 = 1.0
else:
    print "Experiment 3"
    p = 1.5
    lam1 = 1.0
    lam2 = 1.0

print "Experiment "+str(Exp) + ": p = "+str(p)

print "generating the random data..."
A = np.random.normal(0,1,[n,d])

print "normalizing A..."

A = A - sum(A)/d
A = A.dot(np.diag(1/np.sqrt(sum(A**2))))



print"generating true signal..."
k = 10 # number of nonzero groups
width = d/40 # size of each group
minLevel = -5
maxLevel = 5
xtest = np.random.randint(minLevel,maxLevel,k) # constant val of each group, integer between -5 and 5
xtest = np.concatenate([xtest,np.zeros(d/width - k)])
np.random.shuffle(xtest)
Xtst = np.array([np.ones(width)*xtest[i] for i in range(len(xtest))])
xtest = Xtst.reshape(d)


print "generating noise with outliers..."
p_in = 0.9 # probability will be inlier
sigma_out = 5.0 #outlier standard deviation
outlier_mask = 1.0*(np.random.rand(n)>p_in) # selecting which measurments will be outliers w.p. 10%

epsin = np.random.normal(0,1,n) # inlier noise
epsout = np.random.normal(0,sigma_out,n) #outlier noise



print "generating measurements..."
b = A.dot(xtest)+ outlier_mask*epsout + (1-outlier_mask)*epsin



# define the functions used in each algorithm

def theGrad(x):
    # gradient w.r.t. the loss (1/p)*np.linalg.norm(A.dot(x)-b,p)**p
    r = A.dot(x) - b
    return A.T.dot(abs(r)**(p-1)*np.sign(r))

def theSubGrad(x):
    # subgradient of the ell_1 norm
    return np.sign(x)

def theProx(x,rholam):
    # prox of the ell_1 norm
    return fl.proxL1(x,rholam)

def theFunc(x):
    # overall objective function
    return lam1*np.linalg.norm(fl.diffOperator(x),1) + (p**(-1))*np.linalg.norm(A.dot(x)-b,p)**p + lam2*np.linalg.norm(x,1)

def theProxL1star(x,thresh):
    # projection onto the ell_infinity ball
    # this is the prox of the conjugate function of the ell_1 norm and is used in Tseng-pd
    return fl.projLInf(x, thresh)


iter = 2000 # number of iterations, same for each algorithm

print "running projective splitting (ps)..."
Delta = 1e0
gamma = 1e0
rho1_ps = 1e0
rho2_ps = 1e0
rho3_ps = 1e0
rhoIncrease = 1.0
t_ps = time.time()

[fx2_ps,x2_ps,mults_ps,times_ps] = \
    fl.ps(d,iter,rho1_ps,rho2_ps,rho3_ps,theProx,theGrad,theFunc,gamma,lam1,lam2,Delta)
print "ps running time: "+str(time.time()-t_ps)


print "running the subgradient method (subg)..."
alpha_0 = 1e0
r = 1.0

tsubgstart = time.time()
iterSG = 2*iter #subg iterations only require two matrix multiplies by A rather
#than 4 as in ps and tseng-pd, so we allow twice as many total iterations

[x_sg,Fx_sg,times_sg] = fl.subg(d,iterSG,theGrad,theSubGrad,lam1,lam2,alpha_0,r,theFunc)
tsubgend = time.time()
print"subg running time: "+str(tsubgend-tsubgstart)


print "running the proximal subgradient method (prox-sg)"
alpha_0 = 1e0
r = 1.0
t_proxSG_start = time.time()
[x_proxSG,Fx_proxSG,times_proxsg] = fl.proxSubG(d,iterSG,theGrad,theProx,alpha_0,r,lam2,theFunc,lam1,theSubGrad)
t_proxSG_end = time.time()
print "prox SG running time: "+str(t_proxSG_end-t_proxSG_start)


print "running Tseng's method (tseng-pd)"
#### Tseng-PD ####
if Exp==1:
    # p=1.7  case
    gamma = 1e1
elif Exp==2:
    # p=1.3 case
    gamma = 1e2
else:
    # p=1.5 case
    gamma = 1e1

alpha = 1e0
theta = 0.99



t_tg_prod_start = time.time()
[x_tg_prod,F_x_tg_prod,mults_tg_prod,times_tgprod] = fl.tseng_pd(d,iter,theGrad,alpha,theProxL1star,theta,theFunc,lam1,lam2,gamma)
t_tg_prod_end = time.time()
print"Tseng prod running time: "+str(t_tg_prod_end-t_tg_prod_start)


print "Plotting results..."
plotXvals = True
if(plotXvals):
    print "plotting the true and recovered signals"
    plt.plot(xtest) # true signal
    plt.plot(x2_ps) # result returned by ps

    plt.legend(['true signal','ps recovered signal'])
    plt.show()

# estimate the optimal value as the minimum returned by all methods
opt = min(fx2_ps + Fx_sg+F_x_tg_prod+Fx_proxSG)

rawPlots = True
if(rawPlots):
    print "plotting the function values vs number of multiplies"
    plt_range = 100 #print only plot the first plt_range iterations
    plt.plot(mults_ps[0:plt_range],fx2_ps[0:plt_range])
    plt.plot(range(0,2*plt_range+2,2),Fx_sg[0:plt_range+1])
    plt.plot(range(0,2*plt_range+2,2),Fx_proxSG[0:plt_range+1])
    plt.plot(mults_tg_prod[0:plt_range],F_x_tg_prod[0:plt_range])
    plt.plot(opt*np.ones(plt_range))

    plt.legend(['ps', 'subg','prox-sg','tseng-pd'])
    plt.title('function values')
    plt.show()


logPlots = True
if(logPlots):
    print "plotting relative error of function values on a semi-log plot vs number of matrix multiplies"
    plt.semilogy( mults_ps,(fx2_ps-opt)/opt)
    plt.semilogy(mults_tg_prod, (F_x_tg_prod - opt) / opt)
    plt.semilogy( range(0,2*iterSG+2,2),(Fx_sg - opt)/opt) #number of multiplies per iteration is 2 for subg and prox subg
    plt.semilogy( range(0,2*iterSG+2,2),(Fx_proxSG - opt)/opt)

    plt.legend(['ps','tseng-pd','sg','prox-sg'])
    plt.xlabel('number of matrix multiplies')
    plt.ylabel('relative objective optimality gap')
    plt.show()

time_plts = True
if(time_plts):
    print "plotting relative error of function values on a semi-log plot vs cumulative time"
    plt.semilogy(times_ps,(fx2_ps-opt)/opt)
    plt.semilogy(times_tgprod, (F_x_tg_prod - opt) / opt)
    plt.semilogy(times_sg,(Fx_sg - opt)/opt)
    plt.semilogy(times_proxsg,(Fx_proxSG - opt)/opt)

    plt.legend(['ps','tseng-pd','sg','prox-sg'])
    plt.xlabel('running time (s)')
    plt.ylabel('relative objective optimality gap')
    plt.show()
