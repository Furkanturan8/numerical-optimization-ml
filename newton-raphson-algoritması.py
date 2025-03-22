#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 22:33:49 2025

@author: furkanturan
"""

x = 3.5
def f(x):
    return (x - 1) ** 2 * (x - 2) * (x - 3)

def f1(x):  # f(x)'in birinci türevi
    return 2 * (x - 1) * (x - 2) * (x - 3) + (x - 1) ** 2 * ((x - 3) + (x - 2))

def f2(x):  # f(x)'in ikinci türevi
    return (
        2 * ((x - 2) * (x - 3) + (x - 1) * (x - 3) + (x - 1) * (x - 2))
        + 2 * (x - 1) * 2
    )

def dx(f1, f2):
    return -f1/f2

a = f1(x)
b = f2(x)
d = dx(a,b)

iteration = 0
print(iteration, 'x: ',x, ' f: ',f, ' f1: ',f1, ' f2: ',f2, ' dx: ',dx)

while abs(a)>1e-10:
    iteration += 1
    x = x + d
    n = f(x)
    a = f1(x)
    b = f2(x)
    d = dx(a,b)
    print(iteration, 'x: ',x, ' f: ',n, ' f1: ',a, ' f2: ',b, ' dx: ',d)
