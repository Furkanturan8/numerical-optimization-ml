'''

Adım 1: Başlangıç olarak xa ve xb değerlerini öyle bir belirle ki f'(xa).f'(xb) > 0 olsun
Adım 2: xk = xa + (xb-xa)/2 noktasını hesapla
Adım 3: Eğer f'(xk) = 0 veya (xb-xa) < 10^-4 şartını sağlıyorsa algoritmayı sonlandır.
Sağlamıyor ise ve f'(xa).f'(xb) > 0 ise xa <- xk yap eğer sağlamıyorsa xb <- xk yap.
Adım 4: Adım 2 ye git.

'''

def f(x):
    f = (x-1)**2*(x-2)*(x-3)
    return f

def f1(x):
    return 2 * (x - 1) * (x - 2) * (x - 3) + (x - 1) ** 2 * ((x - 3) + (x - 2))

xa = 0.5
xb = 50
iterations = 0


while True:

    xk = xa + (xb-xa)/2
    iterations += 1
    
    print(f"i: {iterations}: xa = {xa:.5f}, xb = {xb:.5f}, xk = {xk:.5f}")

    if (f1(xk) == 0 or (xb-xa) < 1e-4):
        break
    else: 
        if(f1(xa) * f1(xk) > 0):
            xa = xk
        else:
            xb = xk    
            
print(f"Kök yaklaşık olarak: {xa:.5f}")
print(f"Toplam iterasyon sayısı: {iterations}")