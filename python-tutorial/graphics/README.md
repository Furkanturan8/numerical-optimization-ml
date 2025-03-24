# Python Matplotlib Grafik Kütüphanesi

Bu klasör, Python'da grafik çizimi için kullanılan Matplotlib kütüphanesinin temel kullanımını içerir.

## İçerik

1. `example-plots.py`: Temel grafik örnekleri
   - Çizgi grafiği
   - Kontur (eş yükseklik) grafiği
   - 3D yüzey grafiği

## Matplotlib Temel Bileşenleri

### 1. Grafik Tipleri ve Kod Örnekleri

#### a) Çizgi Grafiği (Line Plot)

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y, 'b-', label='sin(x)')  # 'b-': mavi düz çizgi
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title('Sinüs Grafiği')
plt.legend()
plt.grid(True)
plt.show()
```

#### b) Nokta Grafiği (Scatter Plot)

```python
x = np.random.rand(50)
y = np.random.rand(50)

plt.scatter(x, y, c='red', s=100, alpha=0.5)  # c: renk, s: boyut, alpha: şeffaflık
plt.xlabel('x')
plt.ylabel('y')
plt.title('Nokta Grafiği')
plt.show()
```

#### c) Çubuk Grafiği (Bar Plot)

```python
kategoriler = ['A', 'B', 'C', 'D']
değerler = [4, 3, 2, 1]

plt.bar(kategoriler, değerler, color='green')
plt.xlabel('Kategoriler')
plt.ylabel('Değerler')
plt.title('Çubuk Grafiği')
plt.show()
```

#### d) Pasta Grafiği (Pie Chart)

```python
labels = ['Python', 'Java', 'C++', 'JavaScript']
sizes = [45, 30, 15, 10]

plt.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.title('Programlama Dilleri Kullanım Oranı')
plt.show()
```

### 2. Çok Kullanılan Grafik Özellikleri

#### a) Alt Grafikler (Subplots)

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))  # 1 satır, 2 sütun

ax1.plot(x, np.sin(x), 'r-')
ax1.set_title('Sinüs')

ax2.plot(x, np.cos(x), 'b-')
ax2.set_title('Kosinüs')

plt.tight_layout()  # Grafiklerin düzgün yerleşmesi için
plt.show()
```

#### b) Grafik Özelleştirme

```python
plt.figure(figsize=(10, 6))  # Grafik boyutu
plt.plot(x, y,
    color='blue',           # Renk
    linestyle='--',         # Çizgi stili
    linewidth=2,            # Çizgi kalınlığı
    marker='o',            # Nokta stili
    markersize=8,          # Nokta boyutu
    alpha=0.7              # Şeffaflık
)
plt.grid(True, linestyle=':', alpha=0.6)  # Izgara stili
plt.show()
```

#### c) Metin ve Ok Ekleme

```python
plt.plot(x, y)
plt.annotate('Maksimum',
    xy=(2, 1),             # Ok ucu koordinatları
    xytext=(3, 1.5),       # Metin koordinatları
    arrowprops=dict(facecolor='black', shrink=0.05)
)
plt.show()
```

### 3. Özelleştirme Seçenekleri

#### a) Renk Seçenekleri

- Temel renkler: 'b' (blue), 'g' (green), 'r' (red), 'c' (cyan), 'm' (magenta), 'y' (yellow), 'k' (black), 'w' (white)
- HTML renk kodları: '#FF0000' (kırmızı)
- RGB tuple: (1.0, 0.0, 0.0) (kırmızı)

#### b) Çizgi Stilleri

- '-' : düz çizgi
- '--' : kesikli çizgi
- ':' : noktalı çizgi
- '-.' : nokta-çizgi

#### c) Nokta Stilleri

- 'o' : daire
- 's' : kare
- '^' : üçgen
- '\*' : yıldız
- '+' : artı
- 'x' : çarpı

### 4. 3D ve Kontur Grafikleri

#### a) Kontur (Eş Yükseklik) Grafiği

```python
import numpy as np
import matplotlib.pyplot as plt

# Grid oluştur
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)  # 2D koordinat ızgarası oluşturur

# Fonksiyon değerlerini hesapla
Z = X**2 + Y**2  # örnek fonksiyon f(x,y) = x² + y²

# Kontur grafiği çiz
plt.figure(figsize=(10, 8))
contour = plt.contour(X, Y, Z,
    levels=20,           # Kaç farklı seviye çizileceği
    colors='black',      # Kontur çizgi rengi
    linewidths=0.5      # Çizgi kalınlığı
)
plt.clabel(contour, inline=True, fontsize=8)  # Kontur değerlerini göster

# Renkli kontur grafiği
contourf = plt.contourf(X, Y, Z,
    levels=20,
    cmap='viridis'      # Renk haritası
)
plt.colorbar(label='f(x,y) değeri')  # Renk skalası

plt.xlabel('x')
plt.ylabel('y')
plt.title('Kontur Grafiği: f(x,y) = x² + y²')
plt.axis('equal')  # Eksenleri eşit ölçekle
plt.grid(True)
plt.show()
```

Kontur Grafikleri İçin Önemli Parametreler:

- `levels`: Kaç farklı seviye çizileceği
- `colors`: Kontur çizgilerinin rengi
- `linewidths`: Çizgi kalınlıkları
- `cmap`: Renk haritası (contourf için)
- `inline`: Değer etiketlerinin çizgi üzerinde olup olmayacağı

#### b) 3D Yüzey Grafiği

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Grid oluştur
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2  # örnek fonksiyon f(x,y) = x² + y²

# 3D grafik oluştur
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Yüzey grafiği çiz
surf = ax.plot_surface(X, Y, Z,
    cmap='viridis',     # Renk haritası
    linewidth=0,        # Izgara çizgi kalınlığı
    antialiased=True,   # Kenar yumuşatma
    alpha=0.8           # Şeffaflık
)

# Tel kafes (wireframe) ekle
# ax.plot_wireframe(X, Y, Z, color='black', alpha=0.1)

# Kontur grafiğini 3D yüzeyin altına ekle
# offset değeri kontur grafiğinin z ekseni üzerindeki konumunu belirler
ax.contour(X, Y, Z,
    zdir='z',          # Hangi eksene dik olacağı
    offset=-2,         # Kontur grafiğinin z konumu
    cmap='viridis'
)

# Görünüm ayarları
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('3D Yüzey Grafiği: f(x,y) = x² + y²')

# Renk çubuğu ekle
fig.colorbar(surf, label='f(x,y) değeri')

# Görüş açısını ayarla
ax.view_init(elev=30, azim=45)  # elev: yükseklik açısı, azim: yatay açı

plt.show()
```

3D Grafikler İçin Önemli Parametreler:

- `projection='3d'`: 3D grafik oluşturmak için gerekli
- `cmap`: Renk haritası
- `linewidth`: Yüzey ızgara çizgilerinin kalınlığı
- `alpha`: Şeffaflık
- `antialiased`: Kenar yumuşatma
- `view_init()`: Görüş açısı (elev: dikey açı, azim: yatay açı)

#### c) Farklı 3D Grafik Tipleri

1. Yüzey Grafiği: `ax.plot_surface()`
2. Tel Kafes: `ax.plot_wireframe()`
3. Nokta Bulutu: `ax.scatter3D()`
4. Çizgi Grafiği: `ax.plot3D()`

#### d) Yararlı İpuçları

1. Görüş açısını interaktif değiştirmek için:

```python
ax.view_init(elev=None, azim=None)  # None kullanınca fare ile döndürülebilir
```

2. Eksen ölçeklerini eşitlemek için:

```python
ax.set_box_aspect([1,1,1])  # Küp şeklinde görünüm
```

3. Izgara çizgilerini gizlemek için:

```python
ax.grid(False)
```

4. Eksenleri gizlemek için:

```python
ax.set_axis_off()
```

## İpuçları ve Püf Noktaları

1. Grafikleri kaydetme:

```python
plt.savefig('grafik.png', dpi=300, bbox_inches='tight')
```

2. Türkçe karakter desteği:

```python
plt.rcParams['font.family'] = 'DejaVu Sans'
```

3. Grafik stillerini değiştirme:

```python
plt.style.use('seaborn')  # ya da 'ggplot', 'dark_background', vb.
```

4. Renk haritaları (colormap) kullanımı:

```python
plt.scatter(x, y, c=z, cmap='viridis')
plt.colorbar()
```

## Daha Fazla Bilgi

- [Matplotlib Resmi Dokümantasyonu](https://matplotlib.org/stable/contents.html)
- [Matplotlib Örnekler Galerisi](https://matplotlib.org/stable/gallery/index.html)

## Örnekler

Daha detaylı örnekler için `example-plots.py` dosyasına bakınız.
