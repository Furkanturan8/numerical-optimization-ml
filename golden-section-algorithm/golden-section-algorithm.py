#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 23:04:18 2025

@author: furkanturan
"""

import math

def f(x):
    return (x - 1) ** 2 * (x - 2) * (x - 3)

xalt = 2
xust = 3
dx = 0.0000001
alpha = (1+math.sqrt(5))/2
tau = 1-1/alpha

epsilon = dx /(xust-xalt)
N = round(-2.078*math.log(epsilon))

k = 0
x1 = xalt + tau*(xust-xalt); f1 = f(x1);
x2 = xust + tau*(xust-xalt); f2 = f(x2);

for k in range(0,N):
    if f1>f2:
        xalt = x1; x1 = x2; f1 = f2;
        x2 = xust - tau*(xust - xalt); f2 = f(x2);
    else:
        xust = x2; x2 = x1; f2 = f1;
        x1 = xalt + tau*(xust - xalt); f1 = f(x1);

    print(f"{k+1:2d}: x1 = {x1:.5f}, x2 = {x2:.5f}, f1 = {f1:.5f}, f2 = {f2:.5f}")

x = 0.5*(x1+x2)
print('kök: ', x)            
