#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 18:11:10 2025

@author: furkanturan

Davidon-Fletcher-Powell (DFP) Algoritması

DFP algoritması, türev tabanlı bir optimizasyon yöntemidir ve Quasi-Newton sınıfına aittir. 
Bu yöntem, özellikle doğrusal olmayan ve çok değişkenli fonksiyonların minimizasyonunda kullanılır. 
Amacı, Newton yönteminin avantajlarını kullanmakla birlikte, Hessian matrisini doğrudan hesaplamadan 
onun tersini yaklaşık olarak güncelleyerek çözüm sürecini hızlandırmaktır.

Amaç fonksiyonu: f(x) → ℝ
Gradyan: ∇f(x)
Arama yönü: pₖ = -Hₖ ∇f(xₖ)
Adım büyüklüğü: sₖ, genellikle altın oran yöntemiyle belirlenir.

Yeni nokta:
    xₖ₊₁ = xₖ + sₖ * pₖ

Gradyan farkı:
    yₖ = ∇f(xₖ₊₁) - ∇f(xₖ)

x farkı:
    sₖ = xₖ₊₁ - xₖ

Hessian tersinin yaklaşık güncelleme formülü (DFP):
    Hₖ₊₁ = Hₖ + (sₖ sₖᵀ) / (sₖᵀ yₖ) - (Hₖ yₖ yₖᵀ Hₖ) / (yₖᵀ Hₖ yₖ)

Açıklamalar:
- Hₖ: k. iterasyonda Hessian matrisinin tersine dair yaklaşık değer.
- sₖ: x değerlerindeki değişim.
- yₖ: gradyan değerlerindeki değişim.
- Bu formül, ters Hessian matrisini doğrudan güncelleyerek yeni arama yönlerinin daha etkili seçilmesini sağlar.
- DFP yöntemi simetrik ve pozitif tanımlı matrislerin korunmasını hedefler.

Durdurma kriterleri:
- Maksimum iterasyon sayısına ulaşılması
- Fonksiyon değerinde yeterince küçük değişim
- x değerlerinde yeterince küçük değişim
- Gradyan normunun sıfıra yaklaşması (durağan nokta)

DFP, özellikle ikinci türev bilgisine gerek duymadan hızlı ve etkili yakınsama isteyen uygulamalar için tercih edilir.
"""

"""
np.outer(s, s) : İki vektörün dış çarpımını alır.

a = np.array([a1, a2])
b = np.array([b1, b2])

np.outer(a, b) = [[a1*b1, a1*b2],
                  [a2*b1, a2*b2]]
"""


import numpy as np
import math
import matplotlib.pyplot as plt

# Amaç fonksiyonu (f(x))
def f(x): 
    return 3 + (x[0]- 1.5*x[1])**2 + (x[1]-2)**2

# Gradyan fonksiyonu (f'nin türevi, ∇f)
def gradf(x):
    return np.array([2*(x[0] - 1.5*x[1]), -3*(x[0] - 1.5*x[1]) + 2*(x[1] - 2)])

# Altın oranla adım büyüklüğü (line search)
def GSmain(f, xk, pk):
    # f(xk + s * pk) şeklinde tek değişkenli bir fonksiyon oluştur
    def phi(s):  
        return f(xk + s * pk)

    xalt = 0                # alt sınır
    xust = 4                # üst sınır
    dx = 0.00001
    alpha = (1 + math.sqrt(5)) / 2   # altın oran
    tau = 1 - 1 / alpha
    epsilon = dx / (xust - xalt)
    N = round(-2.078 * math.log(epsilon))  # iterasyon sayısı

    # Başlangıç değerleri
    x1 = xalt + tau * (xust - xalt)
    f1 = phi(x1)
    x2 = xust - tau * (xust - xalt)
    f2 = phi(x2)

    # Altın oran arama döngüsü
    for _ in range(N):
        if f1 > f2:
            xalt = x2
            x2 = x1
            f2 = f1
            x1 = xalt + tau * (xust - xalt)
            f1 = phi(x1)
        else:
            xust = x1
            x1 = x2
            f1 = f2
            x2 = xust - tau * (xust - xalt)
            f2 = phi(x2)

    return 0.5 * (x1 + x2)  # minimum tahmini

# Başlangıç noktası
x = np.array([-5.4, 1.7])

# İlk H matrisi: birim matris (2x2)
H = np.identity(2)

# x değerlerini çizmek için sakla
X1 = [x[0]]
X2 = [x[1]]

# Maksimum iterasyon ve hata toleransları
Nmax = 1000
epsilon1 = 1e-9  # fonksiyon değişimi
epsilon2 = 1e-9  # x değişimi
epsilon3 = 1e-9  # gradyan değişimi
k = 0

# Başlık yazdır
print(f"{'k':>2} | {'x[0]':>10} | {'x[1]':>10} | {'f(x)':>10}")
print("-" * 40)

while True:
    g = gradf(x)  # gradyan vektörü (∇f(x))
    
    # Arama yönü: -H * ∇f(x)
    pk = -np.dot(H, g)
    
    # En uygun adım uzunluğunu bul (line search)
    sk = GSmain(f, x, pk)
    
    # Yeni nokta: x + sk * pk
    x_new = x + sk * pk
    
    # Yeni gradyanı hesapla
    g_new = gradf(x_new)

    # DFP formülündeki s ve y vektörleri:
    s = x_new - x         # x değişimi
    y = g_new - g         # gradyan değişimi

    # H * y vektörü
    Hy = np.dot(H, y)

    # DFP güncelleme formülü:
    # H = H + (s s^T)/(s^T y) - (H y y^T H)/(y^T H y)
    H += np.outer(s, s) / np.dot(s, y) - np.outer(Hy, Hy) / np.dot(y, Hy)

    # Değerleri yazdır
    print(f"{k+1:2d} | {x_new[0]:10.6f} | {x_new[1]:10.6f} | {f(x_new):10.6f}")

    # Durdurma kriterleri:
    if k > Nmax:
        print("Max iterasyona ulaşıldı.")
        break
    if abs(f(x_new) - f(x)) < epsilon1:
        print("f(x) artık değişmiyor.")
        break
    if np.linalg.norm(x_new - x) < epsilon2:
        print("x artık değişmiyor.")
        break
    if np.linalg.norm(gradf(x_new)) < epsilon3:
        print("Gradyan sıfıra yakın, durağan nokta.")
        break

    # x değerlerini kaydet (grafik için)
    X1.append(x_new[0])
    X2.append(x_new[1])
    
    # x'i güncelle
    x = x_new
    k += 1

# Sonuçları çiz
plt.plot(X1, X2, marker='o', color='red')
plt.title("DFP Yörüngesi")
plt.xlabel("x[0]")
plt.ylabel("x[1]")
plt.grid(True)
plt.show()
