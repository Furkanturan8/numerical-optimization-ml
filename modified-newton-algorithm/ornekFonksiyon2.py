import numpy as np
import math
#------------------------------------------------------------------------------
def f(xk):
    x1 = xk[0]
    x2 = xk[1]    
    fk = 100*(x2-x1**2)**2 + (1-x1)**2
    return fk

def gradient(xk):
    x1 = xk[0]
    x2 = xk[1]    
    gk = np.array([ -400*x1*(x2-x1**2)-2*(1-x1), 200*(x2-x1**2) ])
    return gk

def hessian(xk):
    x1 = xk[0]
    x2 = xk[1]   
    Hk = np.matrix([[ 1200*x1**2-400*x2+2,  -400*x1], [ -400*x1, 200 ]])
    return Hk    

def error(xk):
    x1 = xk[0]
    x2 = xk[1]    
    ek = np.array([ (10*x2-10*x1**2), (1-x1) ])
    return ek

def jacobian(xk):
    x1 = xk[0]
    x2 = xk[1]    
    Jk = np.matrix([[-20*x1, 10], [-1, 0]])
    return Jk 
#------------------------------------------------------------------------------