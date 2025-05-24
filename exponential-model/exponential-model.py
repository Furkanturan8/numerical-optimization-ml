#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 20:19:34 2025

@author: furkanturan
"""

# Exponential model (validasyona gerek yok)

import numpy as np
import math
from dataset5 import ti, yi

def exponentialIO(t,x):
    yhat = []
    for ti in t:
        toplam = x[0]*math.exp(x[1]*ti)
        yhat.append(toplam)
        
    return yhat
    

def error(xk,ti,yi):
    yhat = exponentialIO(ti,xk)
    return np.array(yi) - np.array(yhat)


def findJacobian(trainingInput, x):
    numOfData = len(trainingInput)
    J = np.matrix(np.zeros((numOfData,2)))
    for i in range(0,numOfData):
        J[i,0] = -math.exp(x[1] * trainingInput[i])
        J[i,1] = -x[0] * trainingInput[i] * math.exp(x[1] * trainingInput[i])
    return J
    
trainingIndices = np.arange(0,len(ti),2)
trainingInput = np.array(ti)[trainingIndices]
trainingOutput = np.array(yi)[trainingIndices]
# validationInput = np.array(ti)[validationIndices] 
# validationIndices = np.arange(1,len(ti),2)
# validationOutput = np.array(yi)[validationIndices] 


MaxIter = 500
epsilon1 = 1e-9
epsilon2 = 1e-9
epsilon3 = 1e-9
mumax = 1e99


x1 = [np.random.random()-0.5]
x2 = [np.random.random()-0.5]
xk = np.array([x1[0],x2[0]])

k = 0; C1 = True;  C2 = True;  C3 = True;  C4 = True; 
# fValidationBest = 1e99; kBest = 0

ek = error(xk,trainingInput,trainingOutput)
fTraining = sum(ek**2)

# eValidation = error(xk,validationInput,validationOutput)
# fValidation = sum(eValidation**2)

FTRA = [fTraining]
# FVAL = [fValidation]

ITERATION = [k]

print('k: ', k, 'x1: ', format(xk[0],'f'), 'x2: ', format(xk[1],'f'), 'f: ', format(fTraining,'f') )

mu = 1; muscal = 10; I = np.identity(2)

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
            x1.append(xk[0])
            x2.append(xk[1])
            loop = False
            print('k: ', k, 'x1: ', format(xk[0],'f'), 'x2: ', format(xk[1],'f'), 'f: ', format(fz,'f') )
        else:        
            mu = mu * muscal
            if mu > mumax:
                loop = False 
                C2 = False
            
    #   eValidation = error(xk, validationInput, validationOutput)
    #   fValidation = sum(eValidation**2)
        
    #    if fValidation < fValidationBest:
    #        fValidationBest = 1*fValidation
    #        xkBest = 1 * xk
    #        kBest = k

    FTRA.append(fTraining)
#   FVAL.append(fValidation)
    ITERATION.append(k)
    
    C1 = k < MaxIter
    C2 = epsilon1 < abs(fTraining - fz)
    C3 = epsilon2 < np.linalg.norm(sk*pk)
    C4 = epsilon3 < np.linalg.norm(gk)
    
#    print('xkBest: ', xkBest)

import matplotlib.pyplot as plt

T = np.arange(min(ti),max(ti),0.1)
yhat = exponentialIO(T,xk)
plt.scatter(ti, yi, color='darkred', marker = 'x')
plt.plot(T,yhat,color='green',linestyle='solid',linewidth=1)
plt.xlabel('ti')
plt.ylabel('yi')
plt.title('Ustel Model', fontstyle='italic')
plt.grid(color = 'green', linestyle='--', linewidth=0.1)
plt.legend(['Ustel Model','Gercek Veri'])
plt.show()

print(xk)








