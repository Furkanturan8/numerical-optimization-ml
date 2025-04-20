import numpy as np
import math
import matplotlib.pyplot as plt

def f(x): 
    return 3 + (x[0]-1.5*x[1])**2 + (x[1]-2)**2

def gradf(x):
    return np.array([2*(x[0]-1.5*x[1]), -3*(x[0]-1.5*x[1]) + 2*(x[1]-2)])

def GSmain(f_line, xk, pk):
    xalt = 0
    xust = 4
    dx = 0.00001
    alpha = (1 + math.sqrt(5)) / 2
    tau = 1 - 1 / alpha

    epsilon = dx / (xust - xalt)
    N = round(-2.078 * math.log(epsilon))

    def f1d(s):  # Tek değişkenli hale getirilmiş fonksiyon
        return f_line(xk + s * pk)

    x1 = xalt + tau * (xust - xalt)
    x2 = xust - tau * (xust - xalt)
    f1 = f1d(x1)
    f2 = f1d(x2)

    for _ in range(N):
        if f1 > f2:
            xalt = x1
            x1 = x2
            f1 = f2
            x2 = xust - tau * (xust - xalt)
            f2 = f1d(x2)
        else:
            xust = x2
            x2 = x1
            f2 = f1
            x1 = xalt + tau * (xust - xalt)
            f1 = f1d(x1)

    return 0.5 * (x1 + x2)

# Adım 1
x = np.array([-5.4, 1.7])
X1 = [x[0]]
X2 = [x[1]]
Nmax = 10000
epsilon1 = 1e-9
epsilon2 = 1e-9
epsilon3 = 1e-9
k = 0

updatedX = np.array([1e10, 1e10])

while True:
    k += 1
    pk = -gradf(x)
    sk = GSmain(f, x, pk)
    x_new = x + sk * pk

    print(f"Iterasyon {k:3d}: x = {x}, f(x) = {f(x):.6f}, |grad(f)| = {np.linalg.norm(gradf(x)):.6e}, step = {sk:.6f}")

    # Güncelleme sonrası durma koşulları
    if k >= Nmax:
        print("🔁 Maksimum iterasyon sayısına ulaşıldı.")
        break
    if abs(f(x_new) - f(x)) < epsilon1:
        print("📉 Fonksiyon değeri yeterince değişmedi.")
        break
    if np.linalg.norm(x_new - x) < epsilon2:
        print("📍 Değişkenler yeterince değişmedi.")
        break
    if np.linalg.norm(gradf(x_new)) < epsilon3:
        print("✅ Durağan noktaya ulaşıldı.")
        break

    x = x_new
    X1.append(x[0])
    X2.append(x[1])

# --------- Grafik
plt.plot(X1, X2, label='İzlenen yol')
plt.scatter(X1, X2, s=5, c='red', label='Noktalar')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Modified Newton Yöntemi ile İyileşme')
plt.legend()
plt.grid(True)
plt.show()
