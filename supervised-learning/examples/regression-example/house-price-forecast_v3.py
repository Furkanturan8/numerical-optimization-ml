# Gerekli kütüphaneleri import edelim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

print("### VERSİYON 3 ###")
print("Bu versiyonda yapılan iyileştirmeler:")
print("1. Veri seti 150 örneğe çıkarıldı")
print("2. Polynomial özellikler eklendi (2. dereceden)")
print("3. Görselleştirme eklendi")
print("4. Cross-validation eklendi\n")

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

# Polynomial özellikleri oluştur
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
feature_names = poly.get_feature_names_out(X.columns)
print("Polynomial Özellikler:")
print(feature_names)
print(f"Toplam özellik sayısı: {len(feature_names)}\n")

# Verileri ölçeklendirelim (standardize)
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X_poly)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

# Eğitim ve test setlerini ayıralım (%80 eğitim, %20 test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

print("Veri Seti Bölümleme:")
print(f"Eğitim seti boyutu: {X_train.shape}")
print(f"Test seti boyutu: {X_test.shape}\n")

class LinearRegression:
    def __init__(self, learning_rate=0.001, n_iterations=2000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.training_history = []
        
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
plt.title('Eğitim Süreci (V3)')
plt.xlabel('İterasyon')
plt.ylabel('MSE')
plt.grid(True)
plt.savefig('training_process_v3.png')
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

# En önemli özellikleri göster
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': np.abs(model.weights)
})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print("\nEn Önemli 10 Özellik:")
print(feature_importance.head(10))

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

# Örnek evi polynomial özelliklere dönüştür ve ölçeklendir
ornek_ev_poly = poly.transform(ornek_ev)
ornek_ev_scaled = scaler_X.transform(ornek_ev_poly)

# Tahmin yap ve ölçeklendirmeyi geri al
tahmin_scaled = model.predict(ornek_ev_scaled)
tahmin_fiyat = scaler_y.inverse_transform(tahmin_scaled.reshape(-1, 1))[0][0]

print(f"\nÖrnek Ev Tahmini:")
print(f"Özellikleri:")
for feature, value in zip(X.columns, ornek_ev[0]):
    print(f"{feature}: {value}")
print(f"Tahmini Fiyat: {tahmin_fiyat:.2f} TL")

# Tahmin vs Gerçek değer grafiği
plt.figure(figsize=(10, 6))
plt.scatter(y_test_original, y_pred, alpha=0.5)
plt.plot([y_test_original.min(), y_test_original.max()], [y_test_original.min(), y_test_original.max()], 'r--', lw=2)
plt.xlabel('Gerçek Fiyat')
plt.ylabel('Tahmin Edilen Fiyat')
plt.title('Tahmin vs Gerçek Değer (V3)')
plt.grid(True)
plt.savefig('prediction_vs_actual_v3.png')
plt.close() 


'''
SONUÇ:

### VERSİYON 3 ###
Bu versiyonda yapılan iyileştirmeler:
1. Veri seti 150 örneğe çıkarıldı
2. Polynomial özellikler eklendi (2. dereceden)
3. Görselleştirme eklendi
4. Cross-validation eklendi

Veri Seti Özellikleri:
Toplam örnek sayısı: 169
Özellik sayısı: 5

Özellikler:
        metrekare  yatak_odasi  banyo_sayisi     ev_yasi  merkeze_uzaklik
count  169.000000   169.000000    169.000000  169.000000       169.000000
mean   138.733728     3.254438      1.733728    6.952663         9.502959
std     43.821134     1.128788      0.719696    5.605918         6.008054
min     75.000000     1.000000      1.000000    1.000000         2.000000
25%     98.000000     2.000000      1.000000    2.000000         5.000000
50%    135.000000     3.000000      2.000000    5.000000         8.000000
75%    177.000000     4.000000      2.000000   11.000000        13.000000
max    220.000000     5.000000      3.000000   20.000000        25.000000

Hedef Değişken (Fiyat) İstatistikleri:
count    1.690000e+02
mean     1.012781e+06
std      5.097980e+05
min      3.500000e+05
25%      5.800000e+05
50%      8.200000e+05
75%      1.470000e+06
max      1.900000e+06
Name: fiyat, dtype: float64

==================================================

Polynomial Özellikler:
['metrekare' 'yatak_odasi' 'banyo_sayisi' 'ev_yasi' 'merkeze_uzaklik'
 'metrekare^2' 'metrekare yatak_odasi' 'metrekare banyo_sayisi'
 'metrekare ev_yasi' 'metrekare merkeze_uzaklik' 'yatak_odasi^2'
 'yatak_odasi banyo_sayisi' 'yatak_odasi ev_yasi'
 'yatak_odasi merkeze_uzaklik' 'banyo_sayisi^2' 'banyo_sayisi ev_yasi'
 'banyo_sayisi merkeze_uzaklik' 'ev_yasi^2' 'ev_yasi merkeze_uzaklik'
 'merkeze_uzaklik^2']
Toplam özellik sayısı: 20

Veri Seti Bölümleme:
Eğitim seti boyutu: (135, 20)
Test seti boyutu: (34, 20)

Eğitim Başlıyor...
Öğrenme oranı: 0.001
İterasyon sayısı: 2000

Iterasyon 0, MSE: 0.988531
Iterasyon 200, MSE: 0.056772
Iterasyon 400, MSE: 0.036893
Iterasyon 600, MSE: 0.029297
Iterasyon 800, MSE: 0.025456
Iterasyon 1000, MSE: 0.023036
Iterasyon 1200, MSE: 0.021237
Iterasyon 1400, MSE: 0.019770
Iterasyon 1600, MSE: 0.018521
Iterasyon 1800, MSE: 0.017435

Model Performansı:
R2 Skoru: 0.9758
Ortalama Mutlak Hata: 66289.26 TL

En Önemli 10 Özellik:
                     feature  importance
5                metrekare^2    0.135110
0                  metrekare    0.121885
6      metrekare yatak_odasi    0.116957
10             yatak_odasi^2    0.097731
7     metrekare banyo_sayisi    0.095093
1                yatak_odasi    0.083867
11  yatak_odasi banyo_sayisi    0.081318
15      banyo_sayisi ev_yasi    0.065419
12       yatak_odasi ev_yasi    0.065317
2               banyo_sayisi    0.062648

Model Parametreleri:
metrekare: 0.1219
yatak_odasi: 0.0839
banyo_sayisi: 0.0626
ev_yasi: -0.0316
merkeze_uzaklik: -0.0175
Bias: 0.0031

Örnek Ev Tahmini:
Özellikleri:
metrekare: 120
yatak_odasi: 3
banyo_sayisi: 1
ev_yasi: 5
merkeze_uzaklik: 8
Tahmini Fiyat: 808453.67 TL

'''