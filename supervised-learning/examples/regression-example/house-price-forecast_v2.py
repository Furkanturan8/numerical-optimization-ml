# Gerekli kütüphaneleri import edelim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # MinMaxScaler yerine StandardScaler kullanacağız
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

print("### VERSİYON 2 ###")
print("Bu versiyonda yapılan iyileştirmeler:")
print("1. StandardScaler kullanıldı (hem X hem y için)")
print("2. Öğrenme oranı 0.001'e düşürüldü")
print("3. İterasyon sayısı 2000'e çıkarıldı")
print("4. Ölçeklendirme işlemleri düzenlendi\n")

# Veri setini yükleyelim
data = pd.read_csv('house_prices.csv')

# Özellikler (features) ve hedef değişkeni (target) ayıralım
X = data[['metrekare', 'yatak_odasi', 'banyo_sayisi', 'ev_yasi', 'merkeze_uzaklik']]
y = data['fiyat']

print("Veri Seti Özellikleri:")
print(f"Toplam örnek sayısı: {len(data)}")
print(f"Özellik sayısı: {len(X.columns)}")
print("\nÖzellikler:")
print(X.describe())
print("\nHedef Değişken (Fiyat) İstatistikleri:")
print(y.describe())
print("\n" + "="*50 + "\n")

# Verileri ölçeklendirelim (standardize)
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

# Eğitim ve test setlerini ayıralım (%80 eğitim, %20 test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

print("Veri Seti Bölümleme:")
print(f"Eğitim seti boyutu: {X_train.shape}")
print(f"Test seti boyutu: {X_test.shape}\n")

# Şimdi Doğrusal Regresyon modelimizi oluşturacağız
class LinearRegression:
    def __init__(self, learning_rate=0.001, n_iterations=2000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.training_history = []  # Eğitim sürecini takip etmek için
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        print("Eğitim Başlıyor...")
        print(f"Öğrenme oranı: {self.learning_rate}")
        print(f"İterasyon sayısı: {self.n_iterations}\n")
        
        # Gradient descent algoritması
        for _ in range(self.n_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias
            
            # Gradyanları hesapla
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)
            
            # Ağırlıkları güncelle
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Her 200 iterasyonda bir MSE hesapla ve yazdır
            if _ % 200 == 0:
                mse = np.mean((y_predicted - y) ** 2)
                self.training_history.append(mse)
                print(f'Iterasyon {_}, MSE: {mse:.6f}')
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Model oluştur ve eğit
model = LinearRegression(learning_rate=0.001, n_iterations=2000)
model.fit(X_train, y_train)

# Eğitim sürecini görselleştir
plt.figure(figsize=(10, 6))
plt.plot(range(0, model.n_iterations, 200), model.training_history)
plt.title('Eğitim Süreci (V2)')
plt.xlabel('İterasyon')
plt.ylabel('MSE')
plt.grid(True)
plt.savefig('training_process_v2.png')
plt.close()

# Test seti üzerinde tahmin yap
y_pred_scaled = model.predict(X_test)

# Ölçeklendirilmiş değerleri geri dönüştür
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

# Model performansını değerlendir
r2 = r2_score(y_test_original, y_pred)
mae = mean_absolute_error(y_test_original, y_pred)

print("\nModel Performansı:")
print(f"R2 Skoru: {r2:.4f}")
print(f"Ortalama Mutlak Hata: {mae:.2f} TL")

# Tahmin vs Gerçek değer grafiği
plt.figure(figsize=(10, 6))
plt.scatter(y_test_original, y_pred, alpha=0.5)
plt.plot([y_test_original.min(), y_test_original.max()], [y_test_original.min(), y_test_original.max()], 'r--', lw=2)
plt.xlabel('Gerçek Fiyat')
plt.ylabel('Tahmin Edilen Fiyat')
plt.title('Tahmin vs Gerçek Değer (V2)')
plt.grid(True)
plt.savefig('prediction_vs_actual_v2.png')
plt.close()

# Öğrenilen parametreleri yazdır
print("\nModel Parametreleri:")
for i, (feature, weight) in enumerate(zip(X.columns, model.weights)):
    print(f"{feature}: {weight:.4f}")
print(f"Bias: {model.bias:.4f}")

# Örnek bir tahmin yapalım
ornek_ev = np.array([[
    120,  # metrekare
    3,    # yatak odası
    1,    # banyo sayısı
    5,    # ev yaşı
    8     # merkeze uzaklık
]])

# Örnek evi ölçeklendir
ornek_ev_scaled = scaler_X.transform(ornek_ev)

# Tahmin yap ve ölçeklendirmeyi geri al
tahmin_scaled = model.predict(ornek_ev_scaled)
tahmin_fiyat = scaler_y.inverse_transform(tahmin_scaled.reshape(-1, 1))[0][0]

print(f"\nÖrnek Ev Tahmini:")
print(f"Özellikleri:")
for feature, value in zip(X.columns, ornek_ev[0]):
    print(f"{feature}: {value}")
print(f"Tahmini Fiyat: {tahmin_fiyat:.2f} TL") 

'''
SONUÇ:

### VERSİYON 2 ###
Bu versiyonda yapılan iyileştirmeler:
1. StandardScaler kullanıldı (hem X hem y için)
2. Öğrenme oranı 0.001'e düşürüldü
3. İterasyon sayısı 2000'e çıkarıldı
4. Ölçeklendirme işlemleri düzenlendi

Veri Seti Özellikleri:
Toplam örnek sayısı: 20
Özellik sayısı: 5

Özellikler:
        metrekare  yatak_odasi  banyo_sayisi    ev_yasi  merkeze_uzaklik
count   20.000000    20.000000     20.000000  20.000000        20.000000
mean   136.000000     3.200000      1.700000   7.200000         9.750000
std     43.546708     1.151658      0.732695   5.643814         6.033895
min     75.000000     1.000000      1.000000   1.000000         2.000000
25%     98.750000     2.000000      1.000000   2.750000         5.750000
50%    135.000000     3.000000      2.000000   5.500000         8.000000
75%    171.250000     4.000000      2.000000  10.250000        12.500000
max    220.000000     5.000000      3.000000  20.000000        25.000000

Hedef Değişken (Fiyat) İstatistikleri:
count    2.000000e+01
mean     9.830000e+05
std      5.039225e+05
min      3.500000e+05
25%      5.750000e+05
50%      8.150000e+05
75%      1.412500e+06
max      1.900000e+06
Name: fiyat, dtype: float64

==================================================

Veri Seti Bölümleme:
Eğitim seti boyutu: (16, 5)
Test seti boyutu: (4, 5)

Eğitim Başlıyor...
Öğrenme oranı: 0.001
İterasyon sayısı: 2000

Iterasyon 0, MSE: 1.130681
Iterasyon 200, MSE: 0.194819
Iterasyon 400, MSE: 0.072775
Iterasyon 600, MSE: 0.054146
Iterasyon 800, MSE: 0.049081
Iterasyon 1000, MSE: 0.046137
Iterasyon 1200, MSE: 0.043791
Iterasyon 1400, MSE: 0.041794
Iterasyon 1600, MSE: 0.040067
Iterasyon 1800, MSE: 0.038565

Model Performansı:
R2 Skoru: -0.6271
Ortalama Mutlak Hata: 115913.52 TL

Model Parametreleri:
metrekare: 0.2780
yatak_odasi: 0.2620
banyo_sayisi: 0.2418
ev_yasi: -0.1210 
merkeze_uzaklik: -0.1084
Bias: 0.0481

Örnek Ev Tahmini:
Özellikleri:
metrekare: 120
yatak_odasi: 3
banyo_sayisi: 1
ev_yasi: 5
merkeze_uzaklik: 8
Tahmini Fiyat: 855416.98 TL

'''
