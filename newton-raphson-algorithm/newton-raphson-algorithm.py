#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 22:47:06 2025

@author: furkanturan
"""

def f(x):
    return (x - 1) ** 2 * (x - 2) * (x - 3)

def f1(x):
    return 2 * (x - 1) * (x - 2) * (x - 3) + (x - 1) ** 2 * ((x - 3) + (x - 2))

def f2(x):
    return 2 * ((x - 2) * (x - 3) + (x - 1) * (x - 3) + (x - 1) * (x - 2)) + 4 * (x - 1)

def newton_raphson(x0, tol=1e-10, max_iter=100):
    x = x0
    for iteration in range(max_iter):
        fx = f(x)
        f1x = f1(x)
        f2x = f2(x)

        if abs(f1x) < tol:  # Eğimin sıfıra yakın olduğu durumda dur
            print("Türevin sıfıra çok yakın olduğu nokta bulundu.")
            break

        delta_x = -f1x / f2x
        x += delta_x

        print(f"i: {iteration + 1}: x = {x:.5f}, f(x) = {fx:.5f}, f'(x) = {f1x:.5f}, f''(x) = {f2x:.5f}, dx = {delta_x:.5f}")

        if abs(delta_x) < tol:  # Hassasiyete ulaşıldığında dur
            break

    return x

# Başlangıç noktası
x0 = 3.5
root = newton_raphson(x0)
print(f"\nYaklaşık kök: {root:.10f}")
