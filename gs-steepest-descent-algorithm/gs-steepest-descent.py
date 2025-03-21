#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 21:47:52 2025

@author: furkanturan
"""

"""
Bu kod Altın Bölme Arama Yöntemi (Golden Section Search, GS) ve
Düzensiz Dik İniş Yöntemi (Steepest Descent) kullanarak bir fonksiyonun minimumunu bulmaya çalışır.

Gradyan İnişi (Gradient Descent) algoritmasıyla 
Altın Oran Arama Yöntemi (Golden Section Search)'i birleştirerek 
en uygun adım boyutunu belirleyip fonksiyonun minimumunu bulmaya çalışıyor. 
Matematiksel optimizasyon problemleri için yaygın bir yaklaşımdır.

Fonksiyonun minimum noktasına ulaşılana kadar iteratif olarak ilerler.
"""

import numpy as np
import math
import matplotlib.pyplot as plt

# Fonksiyonumuz
def f(x): 
    return 3 + (x[0] - 1.5*x[1])**2 + (x[1] - 2)**2

# Gradyan Fonksiyonu
def gradF(x):
    return np.array([2 * (x[0] - 1.5 * x[1]),  -3 * (x[0] - 1.5 * x[1]) + 2 * (x[1] - 2)])

# Altın Oran ile Arama Yöntemi
def GSmain(f, xk, pk):
    xalt = 0
    xust = 1
    dx = 0.00001
    alpha = (1 + math.sqrt(5)) / 2
    tau = 1 - 1 / alpha
    epsilon = dx / (xust - xalt)
    N = round(-2.078 * math.log(epsilon))

    x1 = xalt + tau * (xust - xalt)
    f1 = f(xk + x1 * pk)
    x2 = xust - tau * (xust - xalt)
    f2 = f(xk + x2 * pk)

    for _ in range(N):
        if f1 > f2:
            xalt = x1
            x1 = x2
            f1 = f2
            x2 = xust - tau * (xust - xalt)
            f2 = f(xk + x2 * pk)
        else:
            xust = x2
            x2 = x1
            f2 = f1
            x1 = xalt + tau * (xust - xalt)
            f1 = f(xk + x1 * pk)
    
    return 0.5 * (x1 + x2)

# Başlangıç Değeri
x = np.array([-5.4, 1.7])
x1, x2 = [x[0]], [x[1]]
Nmax = 10000
eps1, eps2, eps3 = 1e-10, 1e-10, 1e-10
k = 0 

updatedx = np.array([1e10, 1e10])
C1 = Nmax < k
C2 = abs(f(updatedx) - f(x)) < eps1 
C3 = np.linalg.norm(updatedx - x) < eps2
C4 = np.linalg.norm(gradF(updatedx)) < eps3

# Gradyan İnişi Döngüsü
while not (C1 or C2 or C3 or C4):
    k += 1
    pk = -gradF(x)
    sk = GSmain(f, x, pk)  # Altın Oran Arama ile adım boyutu belirleme
    x = x + sk * pk

    print(f"k: {k}, sk: {round(sk,4)}, x1: {round(x[0],4)}, x2: {round(x[1],4)}, f: {round(f(x),4)}, ||gradF||: {round(np.linalg.norm(gradF(x)),4)}")

    C1 = Nmax < k
    C2 = abs(f(updatedx) - f(x)) < eps1 
    C3 = np.linalg.norm(updatedx - x) < eps2
    C4 = np.linalg.norm(gradF(updatedx)) < eps3
    updatedx = x.copy()
    
    x1.append(x[0])
    x2.append(x[1])

# Durdurma koşulları
if C1:
    print("...max. iterasyon sayısı aşıldı!")
if C2:
    print("...fonksiyon değişmiyor!")
if C3:
    print("...değişkenler değişmiyor!")
if C4:
    print("...durağan noktaya gelindi!")

# Optimizasyon Yolunu Çizme
plt.plot(x1, x2, label="İterasyon Yolu")
plt.scatter(x1, x2, s=5, c='red', label="Adımlar")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.title("Gradient Descent + Golden Section Search")
plt.show()
