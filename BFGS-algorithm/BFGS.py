#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 18:34:52 2025

@author: furkanturan

Braydon Fletcher Goldfarb Shanno (BFGS) Algoritması

BFGS, türev tabanlı bir optimizasyon algoritmasıdır ve özellikle doğrusal olmayan fonksiyonların
minimizasyonunda kullanılan bir Quasi-Newton yöntemidir. Bu algoritma, Hessian matrisinin 
tersinin (yaklaşık olarak) güncellenmesini sağlar ve bu sayede Newton benzeri yöntemle 
daha hızlı yakınsama elde edilir.

Amaç fonksiyonu: f(x) → ℝ
Gradyan: ∇f(x)
Arama yönü: pₖ = -Mₖ⁻¹ ∇f(xₖ)
Adım büyüklüğü: sₖ, genellikle altın oran (golden section) ile bulunur.

xₖ₊₁ = xₖ + sₖ * pₖ

BFGS güncelleme formülü:
    yₖ = ∇f(xₖ₊₁) - ∇f(xₖ)
    sₖ = xₖ₊₁ - xₖ
    ρₖ = 1 / (yₖᵀ sₖ)

    Mₖ₊₁ = Mₖ + (yₖ yₖᵀ) / (yₖᵀ sₖ) - (Mₖ sₖ sₖᵀ Mₖ) / (sₖᵀ Mₖ sₖ)

Yukarıdaki formülde:
- Mₖ: k. iterasyonda Hessian matrisinin tersi tahmini
- yₖ: gradyandaki değişim
- sₖ: x değerindeki değişim
- pₖ: yön vektörü
- sₖ: adım büyüklüğü (altın oranla belirleniyor)

Algoritma, yakınsama kriterlerinden biri sağlanana kadar tekrarlanır:
- Maksimum iterasyon sayısına ulaşma
- Fonksiyon değerinin değişmemesi
- x değerinin değişmemesi
- Gradyan normunun sıfıra yaklaşması (durağan nokta)
"""

import numpy as np
import math
import matplotlib.pyplot as plt

def f(x): 
    return 3 + (x[0] - 1.5*x[1])**2 + (x[1] - 2)**2

def gradf(x):
    return np.array([2*(x[0] - 1.5*x[1]), -3*(x[0] - 1.5*x[1]) + 2*(x[1] - 2)])

def GSmain(f, xk, pk):
    xbottom = 0
    xtop = 1
    dx = 0.00001
    alpha = (1 + math.sqrt(5)) / 2
    tau = 1 - 1 / alpha
    epsilon = dx / (xtop - xbottom)
    N = round(-2.078 * math.log(epsilon))
    x1 = xbottom + tau * (xtop - xbottom)
    f1 = f(xk + x1 * pk)
    x2 = xtop - tau * (xtop - xbottom)
    f2 = f(xk + x2 * pk)
    
    for _ in range(N):
        if f1 > f2:
            xbottom = x1
            x1 = x2
            f1 = f2
            x2 = xtop - tau * (xtop - xbottom)
            f2 = f(xk + x2 * pk)
        else:
            xtop = x2
            x2 = x1
            f2 = f1
            x1 = xbottom + tau * (xtop - xbottom)
            f1 = f(xk + x1 * pk)
    result = 0.5 * (x1 + x2)
    
    return result # Optimum adım büyüklüğü olarak son aralığın orta noktasını döndür

# --- ADIM 1: Başlangıç ---

# Başlangıç noktasını belirle
x = np.array([-5.4, 1.7])

# Maksimum iterasyon sayısını tanımla
Nmax = 10000

# Sonlandırma kriterleri eşikleri:
eps1 = 1e-10  # Fonksiyon değerindeki değişim için eşik
eps2 = 1e-10  # x değişkenindeki değişim için eşik
eps3 = 1e-10  # Gradyan normu için eşik (durağan nokta)
k = 0  # İterasyon sayacı

# Başlangıç metrik matrisini (M0) birim matris olarak ayarla
I = np.identity(2)
M = np.identity(2)
updatedx = np.array([1e10, 1e10])

# Çizim için iterasyon geçmişini kaydetme listeleri
x1 = [x[0]]
x2 = [x[1]]
C1 = Nmax < k
C2 = abs(f(updatedx) - f(x)) < eps1
C3 = np.linalg.norm(updatedx - x ) < eps2
C4 = np.linalg.norm(gradf(updatedx)) < eps3

# --- ADIM 2: Yinelemeli Süreç ---

while not (C1 or C2 or C3 or C4):
    k += 1
    ozdeger,ozvektor = np.linalg.eigh(M)
    
    if np.min(ozdeger) > 0:
        pk = -np.dot(np.linalg.inv(M), gradf(x))
    else:
        mu = abs(np.min(ozdeger)) + 0.001
        pk = -np.dot((np.linalg.inv(M + mu*I)),gradf(x))
    
    sk = GSmain(f, x, pk)
    prevG = gradf(x).reshape(-1,1)
    
    x = x + sk * pk
    x = np.array(x)
    
    currentG = gradf(x).reshape(-1,1)
    y = (currentG - prevG)
    pk = pk.reshape(-1,1)
    Dx = (sk*pk)
    
    A = np.dot(y,np.matrix.transpose(y)) / np.dot(np.matrix.transpose(y), Dx)
    B = np.dot(np.dot(M,Dx),np.dot(np.matrix.transpose(Dx),M)) / np.dot(np.matrix.transpose(Dx), np.dot(M,Dx))
    M = M + A - B
    
    k += 1  # İterasyon sayacını artır
    
    print("İterasyon:", k, " Adım büyüklüğü:", sk, " x:", np.round(x, 4), " f(x):", np.round(f(x), 4))
    
    C1 = Nmax < k
    C2 = abs(f(updatedx) - f(x)) < eps1
    C3 = np.linalg.norm(updatedx - x) < eps2
    C4 = np.linalg.norm(gradf(updatedx)) < eps3
    
    updatedx = 1*x
    x1.append(x[0])
    x2.append(x[1])

if C1:
    print("maksimum iterasyona ulaşıldı")
if C2:
    print("fonksiyon değişmiyor")
if C3:
    print("değişkenler değişmiyor")
if C4:
    print("durağan noktaya gelindi")
    
# --- ADIM 3: İterasyon Geçmişini Çiz ---

plt.figure(figsize=(8, 6))
plt.plot(x1, x2, marker='o', linestyle='-', markersize=3)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('İterasyon Geçmişi')
plt.grid(True)
plt.show()