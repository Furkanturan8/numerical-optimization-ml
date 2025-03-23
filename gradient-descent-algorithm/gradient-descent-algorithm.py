#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 21:30:53 2025

@author: furkanturan
"""

import numpy as np
import matplotlib.pyplot as plt


# Fonksiyonumuz
def f(x): 
    return 3 + (x[0] - 1.5*x[1])**2 + (x[1] - 2)**2

# Gradyan Fonksiyonu
def gradF(x):
    return np.array([2 * (x[0] - 1.5 * x[1]),  -3 * (x[0] - 1.5 * x[1]) + 2 * (x[1] - 2)])

# Hessian Matrisi
def hessianF(x):
    return np.array([[2, 0], [0, 2]])

# Başlangıç noktası
x = np.array([4, 3])
i = 0

# Optimizasyon sürecindeki noktaları sakla
trajectory = [x.copy()]

# Başlangıç değerini ekrana yazdır
print(f"i: {i}, f(x): {f(x):.5f}")

loop = True

# Gradient Descent döngüsü
while loop:
    i += 1
    x = x - 0.2 * gradF(x)  # Yeni x değerini hesapla
    trajectory.append(x.copy())  # Yeni noktayı kaydet
    normGradf = np.linalg.norm(gradF(x))  # Gradient normu hesapla

    # İterasyon bilgilerini ekrana yazdır
    print(f"i: {i}, f(x): {f(x):.5f}, ||gradF(x)||: {normGradf:.8f}")

    # Durdurma kriteri
    if normGradf < 1e-8:
        loop = False

# Sonuçları yazdır
print(f"x* = {x}")

''' 
ÖNEMLİ BİR NOT: Gradient Descent (Dik İniş) yönteminde Hessian matrisi kullanılmaz. 
Gradient Descent, sadece birinci dereceden türevleri (gradyan) kullanır ve Hessian 
matrisi gibi ikinci dereceden türev bilgisine ihtiyaç duymaz. 

Hessian matrisi, genellikle Newton Yöntemi gibi daha gelişmiş optimizasyon algoritmalarında kullanılır.

Kodumuzda Hessian matrisi, optimizasyon sürecinde değil, sadece bulunan noktanın minimum, 
maksimum veya semer noktası olup olmadığını analiz etmek için kullanılmıştır.
'''

# Hessian matrisi ve özdeğerler
H = hessianF(x)
ozdeger, ozvektor = np.linalg.eig(H)

# Minimum, maksimum veya semer noktası kontrolü
if min(ozdeger) > 0:
    print("x* noktası minimum")
elif max(ozdeger) < 0:
    print("x* noktası maksimum")
else:
    print("x* noktası semer noktası")

# Optimizasyon Yolunu Çizme
trajectory = np.array(trajectory)  # Listeyi numpy dizisine çevir

plt.figure(figsize=(8, 6))
plt.plot(trajectory[:, 0], trajectory[:, 1], 'r.-', label="Gradient Descent Yolu")  # Optimizasyon yolu
plt.scatter(trajectory[:, 0], trajectory[:, 1], s=30, c='blue', label="Adımlar")  # Ara noktalar
plt.scatter(trajectory[0, 0], trajectory[0, 1], s=100, c='red', label="Başlangıç")  # Başlangıç noktası
plt.scatter(x[0], x[1], s=100, c='green', label="Minimum")  # Minimum noktası

plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.title("Gradient Descent Optimizasyon Yolu")
plt.show()
