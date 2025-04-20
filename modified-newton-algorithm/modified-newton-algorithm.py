#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 15:25:59 2025

@author: furkanturan

Algoritma: Değiştirilmiş Newton Algoritması

Adım 1: 
----------------------------------
Bir başlangıç noktası (𝐱 ) ve maksimum iterasyon sayısı (𝑁 )
belirle.
Sonlandırma kriterleri için 𝜀 , 𝜀 ve 𝜀 değerlerini belirle.
𝑘 ← 0

Adım 2:
----------------------------------
𝐱 noktasındaki gradyant vektörünü ∇𝑓(𝐱 ) hesapla
𝐱 noktasındaki Hessian matrisini ∇ 𝑓(𝐱 ) hesapla
Eğer Hessian matrisi pozitif-tanımlı ise ilerleme yönü olarak 𝐩 =
−[∇ 𝑓(𝐱 )] ∇𝑓(𝐱 ) seç.
Eğer Hessian matrisi pozitif-tanımlı değilse o zaman uygun bir matris
ilavesi (𝐄 veya 𝜇𝐈) ile onu pozitif-tanımlı hale getir ve ilerleme yönü
olarak 𝐩 = −[∇ 𝑓(𝐱 ) + 𝜇𝐈] ∇𝑓(𝐱 ) veya 𝐩 = −[∇ 𝑓(𝐱 ) +
𝐄] ∇𝑓(𝐱 ) seç.
Bir-boyutlu optimizasyon ile 𝑓(𝐱 + 𝑠 𝐩 ) değerini minimum yapan
adım-aralığını (𝑠 ) bul.
𝐱 = 𝐱 + 𝑠 𝐩 kuralı ile güncellemeyi yap.
𝑘 ← 𝑘 + 1

Adım 3:
----------------------------------
Aşağıdaki şartlardan herhangi biri sağlanıyorsa algoritmayı bitir,
sağlanmıyorsa Adım 2'ye git.
C1: 𝑁 < 𝑘 maksimum iterasyona ulaşıldı.
C2: |∆𝑓| = |𝑓(𝐱 ) − 𝑓(𝐱 )| < 𝜀 fonksiyon değişmiyor.
C3: |∆𝑥| = |𝐱 − 𝐱 | < 𝜀 değişkenler değişmiyor.
C4: ‖∇𝑓(𝐱 )‖ < 𝜀 yerel minimuma yakınsadı.

"""

import numpy as np
from numpy.linalg import norm, solve, eigvals
from scipy.optimize import line_search

def modified_newton(f, grad_f, hess_f, x0, max_iter=100, eps1=1e-6, eps2=1e-6, eps3=1e-6):
    """
    Değiştirilmiş Newton Algoritması
    
    Parametreler:
    ------------
    f : callable
        Minimize edilecek fonksiyon
    grad_f : callable
        Fonksiyonun gradyanı
    hess_f : callable
        Fonksiyonun Hessian matrisi
    x0 : ndarray
        Başlangıç noktası
    max_iter : int
        Maksimum iterasyon sayısı
    eps1, eps2, eps3 : float
        Sonlandırma kriterleri için tolerans değerleri
    
    Dönüş:
    ------
    x : ndarray
        Bulunan minimum nokta
    f_min : float
        Minimum fonksiyon değeri
    success : bool
        Algoritmanın başarılı olup olmadığı
    iter_count : int
        İterasyon sayısı
    """
    
    x = np.array(x0, dtype=float)
    n = len(x)
    k = 0
    f_prev = f(x)
    
    print("\nDeğiştirilmiş Newton Algoritması başlatılıyor...")
    print("=" * 50)
    print(f"Başlangıç noktası: {x}")
    print(f"Başlangıç fonksiyon değeri: {f_prev:.10f}")
    print("=" * 50)
    
    while k < max_iter:
        print(f"\nİterasyon {k+1}")
        print("-" * 30)
        
        # Gradyan ve Hessian hesapla
        grad = grad_f(x)
        hess = hess_f(x)
        
        print(f"Gradyan norm: {norm(grad):.10f}")
        
        # Gradyan normu kontrolü (C4)
        if norm(grad) < eps3:
            print("\nGradyan normu yeterince küçük - Sonlandırılıyor (C4)")
            return x, f(x), True, k
        
        # Hessian matrisinin pozitif tanımlı olup olmadığını kontrol et
        eigenvals = eigvals(hess)
        if np.all(eigenvals > 0):
            # Hessian pozitif tanımlı
            p = -solve(hess, grad)
            print("Hessian pozitif tanımlı")
        else:
            # Hessian'ı pozitif tanımlı yap
            mu = abs(min(0, np.min(eigenvals))) + 0.1
            p = -solve(hess + mu * np.eye(n), grad)
            print(f"Hessian pozitif tanımlı değil - Düzeltme uygulandı (mu={mu:.6f})")
        
        # Doğru yönde arama
        alpha = line_search(f, grad_f, x, p)[0]
        if alpha is None:
            alpha = 0.1  # Varsayılan adım boyutu
            print("Line search başarısız - Varsayılan adım boyutu kullanıldı")
        else:
            print(f"Optimum adım boyutu: {alpha:.6f}")
            
        # Güncelleme
        x_new = x + alpha * p
        f_new = f(x_new)
        
        print(f"Yeni nokta: {x_new}")
        print(f"Yeni fonksiyon değeri: {f_new:.10f}")
        print(f"Fonksiyon değişimi: {abs(f_new - f_prev):.10f}")
        print(f"Konum değişimi: {norm(x_new - x):.10f}")
        
        # Sonlandırma kriterleri kontrolü
        if abs(f_new - f_prev) < eps1:  # C2
            print("\nFonksiyon değişimi yeterince küçük - Sonlandırılıyor (C2)")
            return x_new, f_new, True, k
        
        if norm(x_new - x) < eps2:  # C3
            print("\nKonum değişimi yeterince küçük - Sonlandırılıyor (C3)")
            return x_new, f_new, True, k
        
        # Değerleri güncelle
        x = x_new
        f_prev = f_new
        k += 1
    
    # Maksimum iterasyona ulaşıldı (C1)
    print("\nMaksimum iterasyon sayısına ulaşıldı - Sonlandırılıyor (C1)")
    return x, f(x), False, k

# Test fonksiyonu örneği (Rosenbrock fonksiyonu)
def rosenbrock(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

def rosenbrock_grad(x):
    return np.array([
        -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0]),
        200 * (x[1] - x[0]**2)
    ])

def rosenbrock_hess(x):
    return np.array([
        [-400 * x[1] + 1200 * x[0]**2 + 2, -400 * x[0]],
        [-400 * x[0], 200]
    ])

# Algoritmanın test edilmesi
if __name__ == "__main__":
    x0 = np.array([-1.0, 1.0])
    result = modified_newton(rosenbrock, rosenbrock_grad, rosenbrock_hess, x0)
    
    print("Sonuç:")
    print(f"Minimum nokta: {result[0]}")
    print(f"Minimum değer: {result[1]:.10f}")
    print(f"Başarılı: {result[2]}")
    print(f"İterasyon sayısı: {result[3]}")



