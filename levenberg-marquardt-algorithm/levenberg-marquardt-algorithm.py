#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 19:18:48 2025

@author: furkanturan
"""#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 19:18:48 2025

@author: furkanturan
"""
import numpy as np
import matplotlib.pyplot as plt

# Hedef fonksiyon
def f(x):
    return 3 + (x[0] - 1.5 * x[1])**2 + (x[1] - 2)**2

# Hata fonksiyonu (e(x))
def error(x):
    e1 = x[0] - 1.5 * x[1]
    e2 = x[1] - 2
    return np.array([e1, e2])

# Jacobian matrisi (de/dx)
def jacobian(x):
    # ∂e1/∂x1 = 1, ∂e1/∂x2 = -1.5
    # ∂e2/∂x1 = 0, ∂e2/∂x2 = 1
    return np.array([
        [1.0, -1.5],
        [0.0, 1.0]
    ])

# Başlangıç noktası
x = np.array([0.0, 0.0])
u = 0.01  # Damping faktörü
max_iter = 100

# Adımların kaydını tutmak için liste
trajectory = [x.copy()]

for k in range(max_iter):
    e = error(x)              # Hata vektörü
    J = jacobian(x)           # Jacobian matrisi

    JT = np.transpose(J)      # J'nin transpozu
    JTJ = np.dot(JT, J)       # Jᵀ * J
    JTe = np.dot(JT, e)       # Jᵀ * e

    # Damping terimi eklenmiş matris
    A = JTJ + u * np.identity(2)

    # Güncelleme yönü
    try:
        dz = -np.linalg.solve(A, JTe)
    except np.linalg.LinAlgError:
        print("Matris terslenemiyor, işlem durduruldu.")
        break

    # Yeni aday nokta
    new_x = x + dz

    # Fonksiyon değeri kontrolü
    if f(new_x) < f(x):
        # İyileşme varsa x güncellenir, u azaltılır
        x = new_x
        u = u / 10
    else:
        # İyileşme yoksa damping artırılır
        u = u * 10

    print(f"{k+1:>2}. iterasyon: x = {x}, f(x) = {f(x)}")

    # Adım kaydını tut
    trajectory.append(x.copy())

    # Konverjans kontrolü
    if np.linalg.norm(dz) < 1e-6:
        print("Konverjans sağlandı.")
        break

# İterasyon ilerleyişini görselleştirmek için (iterasyon noktalarını çizme)
trajectory = np.array(trajectory)
x1_vals = trajectory[:, 0]
x2_vals = trajectory[:, 1]

# Fonksiyon yüzeyini çizmek için
x1 = np.linspace(-1, 4, 100)
x2 = np.linspace(-1, 4, 100)
X1, X2 = np.meshgrid(x1, x2)
Z = 3 + (X1 - 1.5 * X2)**2 + (X2 - 2)**2

# Grafik oluşturma
plt.figure(figsize=(8, 6))

# Kontur çizimi
contours = plt.contour(X1, X2, Z, levels=50, cmap='viridis')
plt.clabel(contours, inline=True, fontsize=8)

# Yolu çiz
plt.plot(x1_vals, x2_vals, marker='o', color='red', label='LM Adımları')
plt.scatter(x1_vals[0], x2_vals[0], color='blue', label='Başlangıç', zorder=5)
plt.scatter(x1_vals[-1], x2_vals[-1], color='green', label='Son Nokta', zorder=5)

# Grafik başlıkları ve etiketler
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Levenberg-Marquardt Yöntemi ile Optimizasyon')
plt.legend()
plt.grid(True)

# Görselleştirmeyi göster
plt.show()
