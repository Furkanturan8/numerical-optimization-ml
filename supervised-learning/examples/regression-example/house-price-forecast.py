# Gerekli kütüphaneleri import edelim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

print("### VERSİYON 1 ###")

# Veri setini yükleyelim
data = pd.read_csv('house_prices.csv')

# Özellikler (features) ve hedef değişkeni (target) ayıralım
X = data[['metrekare', 'yatak_odasi', 'banyo_sayisi', 'ev_yasi', 'merkeze_uzaklik']]
y = data['fiyat']

# Verileri normalize edelim (0-1 aralığına getirelim)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Eğitim ve test setlerini ayıralım (%80 eğitim, %20 test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Şimdi Doğrusal Regresyon modelimizi oluşturacağız
class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.training_history = []
        
    def fit(self, X, y):
        # weights ve bias'ı başlangıçta rastgele değerlerle başlatalım
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent algoritması
        for _ in range(self.n_iterations):
            # İleri yayılım (forward propagation)
            y_predicted = np.dot(X, self.weights) + self.bias
            
            # Gradyanları hesapla
            # dw = (1/n_samples) * X^T * (y_predicted - y)
            # db = (1/n_samples) * sum(y_predicted - y)
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)
            
            # Ağırlıkları güncelle
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Her 100 iterasyonda bir MSE hesapla ve yazdır
            if _ % 100 == 0:
                mse = np.mean((y_predicted - y) ** 2)
                self.training_history.append(mse)
                print(f'Iterasyon {_}, MSE: {mse}')
    
    def predict(self, X):
        # Doğrusal regresyon formülü: y = wx + b
        return np.dot(X, self.weights) + self.bias

# Model oluştur ve eğit
model = LinearRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X_train, y_train)

# Eğitim sürecini görselleştir
plt.figure(figsize=(10, 6))
plt.plot(range(0, model.n_iterations, 100), model.training_history)
plt.title('Eğitim Süreci (V1)')
plt.xlabel('İterasyon')
plt.ylabel('MSE')
plt.grid(True)
plt.savefig('training_process_v1.png')
plt.close()

# Test seti üzerinde tahmin yap
y_pred = model.predict(X_test)

# Model performansını değerlendir
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\nModel Performansı:")
print(f"R2 Skoru: {r2:.4f}")
print(f"Ortalama Mutlak Hata: {mae:.2f} TL")

# Tahmin vs Gerçek değer grafiği
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Gerçek Fiyat')
plt.ylabel('Tahmin Edilen Fiyat')
plt.title('Tahmin vs Gerçek Değer (V1)')
plt.grid(True)
plt.savefig('prediction_vs_actual_v1.png')
plt.close()

# Örnek bir tahmin yapalım
ornek_ev = np.array([[
    120,  # metrekare
    3,    # yatak odası
    1,    # banyo sayısı
    5,    # ev yaşı
    8     # merkeze uzaklık
]])

# Örnek evi normalize et
ornek_ev_normalized = scaler.transform(ornek_ev)

# Tahmin yap
tahmin_fiyat = model.predict(ornek_ev_normalized)[0]
print(f"\nÖrnek Ev Tahmini:")
print(f"Özellikleri: {ornek_ev[0]}")
print(f"Tahmini Fiyat: {tahmin_fiyat:.2f} TL")


'''
SONUÇ:

Iterasyon 0, MSE: 1396581250000.0
Iterasyon 100, MSE: 74637117561.00044
Iterasyon 200, MSE: 27753055841.179016
Iterasyon 300, MSE: 16646219552.809319
Iterasyon 400, MSE: 11809471513.950726
Iterasyon 500, MSE: 9617468219.913631
Iterasyon 600, MSE: 8613757648.875061
Iterasyon 700, MSE: 8145446066.257021
Iterasyon 800, MSE: 7918465302.839788
Iterasyon 900, MSE: 7800337848.83157

Model Performansı:
R2 Skoru: -0.3657
Ortalama Mutlak Hata: 87908.37 TL

Örnek Ev Tahmini:
Özellikleri: [120   3   1   5   8]
Tahmini Fiyat: 786938.46 TL

'''