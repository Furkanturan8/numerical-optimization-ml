#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 22:34:09 2025

@author: furkanturan
"""

# TODO: TEKRAR BAK!

import numpy as np 
import math
from dataset6 import ti,yi 
import matplotlib.pyplot as plt

def tanh(x): 
    return (math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x)) 

# --------------------------

def hyperbolicIO(t,x):
    S = int((len(x)-1)/3)
    # 3s+1 parametre sayısı biz parametre sayısını biliyoruz.
    # Buradan node sayısını bulacağız s'yi(node old.dolayı) yalnız bırakalım. 
    # 19 parametre ise 3s+1=19 s = 6 node vardır 
    yhat = []
    for ti in t:
        toplam = x[3*S] # son eleman olan X3s+1 i atadık
        for j in range(0,S):
            toplam += x[2 * S + j] * tanh(x[j] * ti + x[S+j]) 
        yhat.append(toplam)
        
    return yhat
  
# --------------------------

def error(xk,ti,yi):
    yhat = hyperbolicIO(ti,xk)
    return np.array(yi) - np.array(yhat)
 
# --------------------------
    
def findJacobian(t, x): # t = trainingInputs
    S = int((len(x)-1)/3)
    numOfData = len(t)
    J = np.matrix(np.zeros((numOfData,3*S+1)))
    for i in range(0,numOfData):
        for j in range(0,S):
            J[i,j] = -x[j+2*S]*t[i]*(1-tanh(x[j]*t[i] + x[j+S])**2)
        for j in range(S,2*S):
            J[i,j] = -x[j+S]*(1-tanh(x[j-S]*t[i] + x[j])**2)
        for j in range(2*S,3*S):
            J[i,j] = -tanh(x[j-2*S]*t[i]*x[j-S])        
        J[i,3*S] = -1 # 4.terim    
    return J

# --------------------------

def plotResult(ti,yi,xkBest):
    S = int((len(xkBest)-1)/3)
    T = np.arange(min(ti),max(ti),0.1)
    yhat = hyperbolicIO(T, xkBest)
    plt.scatter(ti, yi, color='darkred', marker = 'x')
    plt.plot(T,yhat,color='green',linestyle='solid',linewidth=1)
    plt.xlabel('ti')
    plt.ylabel('yi')
    plt.title(str(S)+'- Düğümlü Üstel Model | FV:'+ str(fValidation), fontstyle='italic')
    plt.grid(color = 'green', linestyle='--', linewidth=0.1)
    plt.legend(['Üstel Model','Gerçek Veri'])
    plt.show()
    
# --------------------------
    
trainingIndices = np.arange(0,len(ti),2)
trainingInput = np.array(ti)[trainingIndices]
trainingOutput = np.array(yi)[trainingIndices]
validationIndices = np.arange(1,len(ti),2)
validationInput = np.array(ti)[validationIndices] 
validationOutput = np.array(yi)[validationIndices] 

# --------------------------

MaxIter = 1000
epsilon1 = 1e-9
epsilon2 = 1e-9
epsilon3 = 1e-9
mumax = 1e99
NODEmax = int((len(trainingInput)-1)/3)


NODE = []; FV = []; globalBest = 1e10; SBest = 0;

for S in range(2,NODEmax):
    xk = np.random.random(3*S+1)-0.5

    k = 0; C1 = True;  C2 = True;  C3 = True;  C4 = True; 
    fValidationBest = 1e99; kBest = 0

    
    ek = error(xk,trainingInput,trainingOutput)
    fTraining = sum(ek**2)
    
    eValidation = error(xk,validationInput,validationOutput)
    fValidation = sum(eValidation**2)
    
    FTRA = [fTraining]
    FVAL = [fValidation]
    
    ITERATION = [k]
    
    print('k: ', k, 'x1: ', format(xk[0],'f'), 'x2: ', format(xk[1],'f'), 'f: ', format(fTraining,'f') )
    
    mu = 1; muscal = 10; I = np.identity(3*S+1)
    
    while C1 & C2 & C3 & C4:
        ek = error(xk, trainingInput, trainingOutput)
        Jk = findJacobian(trainingInput, xk)
        gk = np.array((2 * Jk.transpose().dot(ek)).tolist()[0])
        Hk = 2*Jk.transpose().dot(Jk) + 1e-8 * I
        fTraining = sum(ek**2)
        sk = 1
        loop = True
        while loop:
            zk = -np.linalg.inv(Hk+mu*I).dot(gk)
            zk = np.array(zk.tolist()[0])
            ez = error(xk + sk * zk, trainingInput, trainingOutput)
            fz = sum(ez**2)
            
            if fz < fTraining:
                pk = 1*zk
                mu = mu/muscal
                k += 1
                xk = xk + sk * pk
                loop = False
                print('k: ', k, 'x1: ', format(xk[0],'f'), 'x2: ', format(xk[1],'f'), 'f: ', format(fz,'f') )
            else:        
                mu = mu * muscal
                if mu > mumax:
                    loop = False 
                    C2 = False
                
        eValidation = error(xk, validationInput, validationOutput)
        fValidation = sum(eValidation**2)
            
        if fValidation < fValidationBest:
            fValidationBest = 1*fValidation
            xkBest = 1 * xk
            kBest = k
    
        FTRA.append(fTraining)
        FVAL.append(fValidation)
        ITERATION.append(k)
        
        C1 = k < MaxIter
        C2 = epsilon1 < abs(fTraining - fz)
        C3 = epsilon2 < np.linalg.norm(sk*pk)
        C4 = epsilon3 < np.linalg.norm(gk)
        
    plotResult(ti, yi, xkBest)  
    NODE.append(S)
    FV.append(fValidationBest)
    
    if fValidationBest < globalBest:
        globalBest = 1*fValidationBest
        SBest = S
        
    print('Düğüm Sayısı: ', S, 'FValbest: ',fValidationBest, 'GlobalFvalBest: ',globalBest)

# --------------------------

plt.bar(NODE, FV, color='orange', width=0.4, linestyle='solid', linewidth=1)
plt.bar(SBest, globalBest, color='blue', width=0.4, linestyle='solid',linewidth=1)
plt.axvline(x = kBest, color='b', linestyle='dashed',linewidth=1)
plt.xlabel('RBF Sayısı')
plt.ylabel('Validation Performansı')
plt.title('RBF Modeli Validation Performansı', fontstyle='italic')
plt.grid(color = 'green', linestyle='--', linewidth=0.1)
plt.legend(['training','validation'])
plt.show()

print(SBest)





