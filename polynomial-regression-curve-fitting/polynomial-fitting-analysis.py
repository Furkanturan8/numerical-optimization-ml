"""
Polinom Regresyon Analizi
-------------------------
Bu program, verilen veri noktalarına en iyi uyan polinom modelini bulmayı amaçlar.
Farklı derece polinomları deneyerek en iyi modeli seçer.

Kullanılan yöntemler:
1. En Küçük Kareler Yöntemi (Least Squares)
2. K-fold benzeri validation (Çapraz doğrulama)
3. Model karşılaştırma ve seçimi
"""

import numpy as np
from dataset2 import ti,yi  # ti: zaman verileri, yi: çıktı verileri
import matplotlib.pyplot as plt

# -------------------------------------------------

def polinomIO(t, x):
    """
    Verilen polinom katsayılarına göre tahmin değerlerini hesaplar
    
    Parametreler:
    t: Giriş değerleri (zaman noktaları)
    x: Polinom katsayıları
    
    Dönüş:
    ythat: Tahmin edilen y değerleri
    """
    ythat = []
    for ti in t:
        toplam = 0
        for i in range(0,len(x)):
            toplam += x[i] * (ti ** i)  # Polinomun her terimini hesapla: x0 + x1*t + x2*t^2 + ...
        ythat.append(toplam)
    return ythat

# -------------------------------------------------

def findx(ti, yi, polinom_derece):
    """
    En küçük kareler yöntemi ile polinom katsayılarını bulur
    
    Parametreler:
    ti: Giriş değerleri
    yi: Gerçek çıkış değerleri
    polinom_derece: Oluşturulacak polinomun derecesi
    
    Dönüş:
    x: Bulunan polinom katsayıları
    """
    numOfData = len(ti)
    # Jacobian matrisini oluştur
    J = -np.ones((numOfData, 1))
    for n in range(1, polinom_derece+1):
        # Her polinom derecesi için Jacobian matrisini genişlet
        J = np.hstack((J, -np.ones((numOfData, 1))*np.array(ti).reshape(numOfData,1)**n))
    
    # En küçük kareler çözümü: x = -(J^T * J)^(-1) * J^T * y
    A = np.linalg.inv(J.transpose().dot(J))
    B = J.transpose().dot(yi)
    x = -A.dot(B)
    return x


# -------------------------------------------------


def plotresult(ti,yi,x,fvalidation):
    """
    Sonuçları görselleştirir: Gerçek veri noktaları ve tahmin edilen polinom eğrisi
    """
    T = np.arange(min(ti),max(ti),0.1)  # Sürekli bir eğri için daha fazla nokta oluştur
    yhat = polinomIO(T,x)
    plt.figure(figsize=(10, 6))
    plt.scatter(ti,yi,color="darkred",marker='x')
    plt.plot(T,yhat,color="green",linestyle='solid',linewidth=1)
    plt.xlabel("ti")
    plt.ylabel("yi")
    plt.title(str(len(x)-1)+" dereceden polinom modeli | FV: "+str(fvalidation), fontstyle="italic")
    plt.legend(["Polinom Modeli","Gerçek Veri"])
    plt.grid(True, alpha=0.3)
    plt.show()

# -------------------------------------------------

# Veriyi eğitim ve doğrulama setlerine ayır
# Çift indeksli veriler eğitim için, tek indeksli veriler doğrulama için kullanılır
trainingIndices = np.arange(0,len(ti),2)
trainingInput = np.array(ti)[trainingIndices]
trainingOutput = np.array(yi)[trainingIndices]

validationIndices = np.arange(1,len(ti),2)
validationInput = np.array(ti)[validationIndices]
validationOutput = np.array(yi)[validationIndices]


# -------------------------------------------------


# Her polinom derecesi için model oluştur ve performansını değerlendir
PD = []  # Polinom dereceleri
FV = []  # Validation performans değerleri
for polinom_derece in range(1,10):
    # Polinom katsayılarını bul
    x = findx(trainingInput,trainingOutput,polinom_derece)
    
    # Doğrulama seti üzerinde tahminler yap
    yhat = polinomIO(validationInput,x)
    
    # Hata hesapla (Mean Squared Error - MSE)
    e = np.array(validationOutput)-np.array(yhat)
    fvalidation = sum(e**2)
    
    # Sonuçları kaydet
    PD.append(polinom_derece)
    FV.append(fvalidation)
    print(f"Polinom Derecesi: {polinom_derece}, FV: {fvalidation}")
    plotresult(ti,yi,x,fvalidation)


# -------------------------------------------------


# Son olarak tüm modellerin performans karşılaştırmasını göster
plt.figure(figsize=(12, 6))

# Normal ölçekli grafik
plt.subplot(121)
plt.bar(PD,FV,color='darkred')
plt.xlabel('Polinom derecesi')
plt.ylabel('Validation performansı')
plt.title('Polinom modeli (Normal ölçek)', fontstyle='italic')
plt.grid(color='green', linestyle='--', linewidth=0.1, alpha=0.3)

# Logaritmik ölçekli grafik (küçük değerleri daha iyi görebilmek için)
plt.subplot(122)
plt.bar(PD,FV,color='darkred')
plt.yscale('log')
plt.xlabel('Polinom derecesi')
plt.ylabel('Validation performansı (log ölçek)')
plt.title('Polinom modeli (Logaritmik ölçek)', fontstyle='italic')
plt.grid(color='green', linestyle='--', linewidth=0.1, alpha=0.3)

plt.tight_layout()
plt.show()