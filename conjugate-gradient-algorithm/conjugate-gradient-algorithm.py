"""
Created on Sun Mar 24 01:53:22 2025

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

# Grafik için grid oluştur
x1 = np.linspace(-5, 5, 100)
x2 = np.linspace(-5, 5, 100)
X1, X2 = np.meshgrid(x1, x2)
Z = 3 + (X1 - 1.5*X2)**2 + (X2 - 2)**2

# Kontur grafiği
plt.figure(figsize=(10, 8))
contour = plt.contour(X1, X2, Z, levels=20)
plt.clabel(contour, inline=True, fontsize=8)
plt.xlabel('x₁')
plt.ylabel('x₂')
plt.title('Konjuge Gradyan Algoritması İlerleyişi')
plt.grid(True)

# Algoritma ve nokta takibi
k = 0
eps1, eps2, eps3 = 1e-10, 1e-10, 1e-10
Nmax = 1000
x = np.array([-4.5, -3.5])  # Başlangıç noktası
x_points = [x.copy()]  # Noktaları saklamak için liste
gradf = gradF(x)
gradf_prev = None  # Önceki gradyan değerini saklamak için
pk = -gradf  # İlk arama yönü


while k < Nmax: # C1 sonlandırma kriteri
    if k != 0:
        Beta = np.dot(gradf, gradf) / np.dot(gradf_prev, gradf_prev)
        pk = -gradf + Beta * pk

    # Önceki değerleri sakla
    x_prev = x.copy()
    f_prev = f(x_prev)
    gradf_prev = gradf.copy()
    
    # Yeni x'i hesapla (burada doğru adım boyutu hesaplanmalı)
    sk = 0.01  # Bu değer optimize edilmeli
    x = x + sk * pk    
    x_points.append(x.copy())  # Yeni noktayı listeye ekle

    # Yeni gradyanı hesapla
    gradf = gradF(x)
    
    # Sonlandırma kriterleri C2,C3,C4
    deltaF = abs(f(x) - f_prev)
    deltaX = np.linalg.norm(x - x_prev)
    grad_norm = np.linalg.norm(gradf)  
    
    # Her iterasyondaki değerleri yazdır
    print(f"İterasyon {k}: x = [{x[0]:.4f}, {x[1]:.4f}], f(x) = {f(x):.4f}, ||∇f|| = {grad_norm:.4f}, |Δf| = {deltaF:.4f}, ||Δx|| = {deltaX:.4f}")

    if (deltaF < eps1 and deltaX < eps2 and grad_norm < eps3):
        print("\nSonlandırma kriterleri sağlandı!")
        break
    
    k += 1

# Noktaları grafiğe ekle
x_points = np.array(x_points)
plt.plot(x_points[:, 0], x_points[:, 1], 'r.-', label='Algoritma İlerleyişi', markersize=2)
plt.plot(x_points[0, 0], x_points[0, 1], 'go', label='Başlangıç Noktası')
plt.plot(3, 2, 'r*', markersize=15, label='Minimum (3,2)')
plt.legend()
plt.show()

print(f"\nSonuç:")
print(f"Son nokta: [{x[0]:.4f}, {x[1]:.4f}]")
print(f"Minimum değer: {f(x):.4f}")
print(f"İterasyon sayısı: {k}")
    







    