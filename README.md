# Sayısal Optimizasyon Algoritmaları

Bu repo, çeşitli sayısal optimizasyon algoritmalarının Python implementasyonlarını içermektedir. Her bir algoritma, belirli bir optimizasyon problemini çözmek için tasarlanmıştır.

## İçindekiler

1. [Bisection (İkiye Bölme) Algoritması](#bisection-algoritması)
2. [Newton-Raphson Algoritması](#newton-raphson-algoritması)
3. [Golden Section (Altın Oran) Algoritması](#golden-section-algoritması)
4. [GS Steepest Descent (Altın Oran ile En Dik İniş) Algoritması](#gs-steepest-descent-algoritması)
5. [Gradient Descent (Gradyan İniş) Algoritması](#gradient-descent-algoritması)
6. [Conjugate Gradient (Konjuge Gradyan) Algoritması](#conjugate-gradient-algoritması)

## Bisection Algoritması

Bisection algoritması, bir fonksiyonun kökünü bulmak için kullanılan temel bir sayısal yöntemdir. Algoritma, sürekli bir fonksiyonun işaret değiştirdiği bir aralıkta çalışır.

### Çalışma Prensibi:

1. Başlangıçta f'(xa) ve f'(xb) değerleri zıt işaretli olan iki nokta seçilir
2. Aralığın orta noktası (xk) hesaplanır
3. f'(xk) = 0 veya aralık genişliği belirlenen tolerans değerinden küçük olana kadar:
   - Orta nokta hesaplanır
   - Hangi yarım aralıkta kök olduğu belirlenir
   - Aralık güncellenir

### Algoritma:

```python
1. # Başlangıç değerlerini belirle:
   xa = başlangıç noktası
   xb = bitiş noktası
   f'(xa) * f'(xb) < 0 # olmak şartıyla...

2. # While döngüsü başlat:
   xk = xa + (xb-xa)/2

   if (f'(xk) == 0 or (xb-xa) < tolerans):
       return xk  # Kök bulundu

   if (f'(xa) * f'(xk) > 0):
       xa = xk   # Sol yarıyı ele
   else:
       xb = xk   # Sağ yarıyı ele
```

Uygulamamızda (x-1)²(x-2)(x-3) fonksiyonu üzerinde çalışılmıştır.

## Newton-Raphson Algoritması

Newton-Raphson metodu, fonksiyonların köklerini bulmak için kullanılan güçlü bir iteratif yöntemdir. Birinci ve ikinci türevleri kullanarak hızlı yakınsama sağlar.

### Çalışma Prensibi:

1. Başlangıç noktası seçilir
2. Her iterasyonda:
   - Fonksiyon değeri, birinci ve ikinci türevler hesaplanır
   - x = x - f'(x)/f''(x) formülü ile yeni nokta bulunur
3. Belirlenen hassasiyete ulaşılana kadar devam edilir

### Algoritma:

```python
1. Başlangıç değerlerini belirle:
   x = x0
   tolerans = hassasiyet değeri
   max_iter = maksimum iterasyon sayısı

2. For i in range(max_iter):
   f1x = f'(x)   # Birinci türev
   f2x = f''(x)  # İkinci türev

   if |f1x| < tolerans:
       return x   # Türevin sıfır olduğu nokta bulundu

   delta_x = -f1x / f2x
   x = x + delta_x

   if |delta_x| < tolerans:
       return x   # Yakınsama sağlandı
```

## Golden Section Algoritması

Golden Section algoritması, tek değişkenli fonksiyonların minimumunu bulmak için kullanılan bir optimizasyon yöntemidir. Altın oran (≈ 1.618) kullanılarak arama aralığını sistematik olarak daraltır.

### Çalışma Prensibi:

1. Başlangıç aralığı [xalt, xust] belirlenir
2. Altın oran kullanılarak iki iç nokta (x1, x2) hesaplanır
3. Bu noktalardaki fonksiyon değerleri karşılaştırılır
4. Minimum değerin bulunabileceği alt aralık seçilir
5. İşlem, belirlenen hassasiyete ulaşılana kadar tekrarlanır

### Algoritma:

```python
1. Başlangıç değerlerini belirle:
   xalt = alt sınır
   xust = üst sınır
   dXson = 0.0000001 # ΔXson değerini belirle
   α = (1 + √5)/2  # Altın oran
   τ = 1 - 1/α     # Tau değeri
   e = dXson/(xust-xalt) # tolerans
   N = round(-2.078*math.log(epsilon))  # İterasyon sayısı belirlenir
   k = 0

2. İç noktaları hesapla:
   x1 = xalt + τ*(xust-xalt)
   x2 = xust - τ*(xust-xalt)
   f1 = f(x1)
   f2 = f(x2)

3. While k < N: # hassasiyet kriteri sağlanana kadar
   if f1 > f2:
       xalt = x1
       x1 = x2
       f1 = f2
       x2 = xust - τ*(xust-xalt)
       f2 = f(x2)
       k += 1
   else:
       xust = x2
       x2 = x1
       f2 = f1
       x1 = xalt + τ*(xust-xalt)
       f1 = f(x1)
       k += 1
```

## GS Steepest Descent Algoritması

Bu algoritma, Gradient Descent ve Golden Section yöntemlerinin birleşimidir. Gradient Descent'in en büyük problemlerinden biri olan adım boyutu seçimini, Golden Section Search ile optimize eder.

### Çalışma Prensibi:

1. Başlangıç noktası seçilir
2. Her iterasyonda:
   - Gradyan vektörü hesaplanır
   - Golden Section Search ile optimal adım boyutu belirlenir
   - Nokta güncellenir
3. Durma kriterleri sağlanana kadar devam edilir

### Algoritma:

```python
1. Başlangıç değerlerini belirle:
   x = başlangıç noktası
   ε1, ε2, ε3 = hassasiyet değerleri

2. While (durma kriterleri sağlanana kadar):
   pk = -∇f(x)  # Arama yönü
   sk = GoldenSection(f, x, pk)  # Optimal adım boyutu
   x_new = x + sk * pk

   if (|f(x_new) - f(x)| < ε1 or
       ||x_new - x|| < ε2 or
       ||∇f(x_new)|| < ε3):
       return x_new

   x = x_new
```

## Gradient Descent (Dik iniş) Algoritması

Gradient Descent, çok değişkenli fonksiyonların yerel minimumlarını bulmak için kullanılan iteratif bir optimizasyon algoritmasıdır.

### Çalışma Prensibi:

1. Başlangıç noktası seçilir
2. Her iterasyonda:
   - Gradyan vektörü hesaplanır
   - Sabit bir öğrenme oranı (0.2) ile çarpılır
   - Mevcut noktadan bu değer çıkarılır
3. Gradyan normu belirlenen tolerans değerinden küçük olana kadar devam edilir

### Algoritma:

```python
1. Başlangıç değerlerini belirle:
   x = başlangıç noktası
   α = öğrenme oranı (0.2)
   ε = hassasiyet değeri

2. While ||∇f(x)|| > ε:
   gradyan = ∇f(x)
   x = x - α * gradyan

3. Return x  # Minimum nokta
```

Uygulamamızda 3 + (x₁ - 1.5x₂)² + (x₂ - 2)² fonksiyonu optimize edilmiştir.

## Conjugate Gradient (Konjuge Gradyan) Algoritması

Konjuge gradyan algoritması, özellikle büyük ölçekli optimizasyon problemlerinde etkili olan iteratif bir yöntemdir. Gradyan iniş yönteminin geliştirilmiş bir versiyonudur ve her iterasyonda önceki arama yönünü de hesaba katar.

### Çalışma Prensibi:

1. Başlangıç noktası seçilir
2. İlk arama yönü negatif gradyan olarak belirlenir
3. Her iterasyonda:
   - Beta katsayısı hesaplanır (Fletcher-Reeves formülü)
   - Yeni arama yönü belirlenir
   - Adım boyutu ile yeni nokta hesaplanır
4. Durma kriterleri sağlanana kadar devam edilir

### Algoritma:

```python
1. Başlangıç değerlerini belirle:
   x = başlangıç noktası
   ε₁, ε₂, ε₃ = hassasiyet değerleri
   Nmax = maksimum iterasyon sayısı
   k = 0

2. İlk gradyanı hesapla:
   gradf = ∇f(x)
   pk = -gradf  # İlk arama yönü

3. While k < Nmax:
   if k > 0:
      # Fletcher-Reeves formülü
      Beta = ||∇f(x_k)||² / ||∇f(x_{k-1})||²
      pk = -gradf + Beta * pk

   # Önceki değerleri sakla
   x_prev = x
   gradf_prev = gradf

   # Yeni noktayı hesapla
   sk = adım boyutu
   x = x + sk * pk

   # Yeni gradyanı hesapla
   gradf = ∇f(x)

   # Sonlandırma kriterleri
   if (|f(x) - f(x_prev)| < ε₁ and
       ||x - x_prev|| < ε₂ and
       ||∇f(x)|| < ε₃):
       break

   k += 1
```

### Önemli Parametreler:

1. **Sonlandırma Kriterleri**:

   - C1: k < Nmax (maksimum iterasyon)
   - C2: |f(x\_{k+1}) - f(x_k)| < ε₁ (fonksiyon değişimi)
   - C3: ||x\_{k+1} - x_k|| < ε₂ (konum değişimi)
   - C4: ||∇f(x\_{k+1})|| < ε₃ (gradyan normu)

2. **Beta Katsayısı**:

   - Fletcher-Reeves formülü: β = ||∇f(x*k)||² / ||∇f(x*{k-1})||²
   - Bu katsayı, yeni arama yönünün hesaplanmasında kullanılır

3. **Adım Boyutu**:
   - Sabit bir değer veya line search ile optimize edilebilir
   - Çok küçük: yavaş yakınsama
   - Çok büyük: ıraksama riski

### Avantajları:

1. Gradyan iniş yöntemine göre daha hızlı yakınsama
2. Özellikle kuadratik fonksiyonlarda etkili
3. Hafıza kullanımı açısından verimli

### Dezavantajları:

1. Beta hesaplaması için ekstra işlem gerektirir
2. Adım boyutu seçimi hala önemli bir faktör
3. Non-kuadratik fonksiyonlarda performans değişkenlik gösterebilir

## Gereksinimler

- Python 3.x
- NumPy
- Matplotlib

## Kullanım

Her bir algoritma kendi klasöründe bulunmaktadır ve bağımsız olarak çalıştırılabilir. Algoritmaların çalışması görselleştirilmiş ve iterasyon adımları konsola yazdırılmıştır.

## Görselleştirmeler

Her algoritmanın klasöründe, algoritmanın çalışmasını gösteren PNG formatında grafikler bulunmaktadır. Bu grafikler, algoritmaların yakınsama davranışını ve optimizasyon sürecini görsel olarak anlamak için faydalıdır.
