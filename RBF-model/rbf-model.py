#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 23:33:51 2025

@author: furkanturan
"""

"""
Radyal Tabanlı Fonksiyon (RBF) Modeli
------------------------------------
Bu program, verilen veri noktalarına RBF ağı kullanarak bir model uydurmayı amaçlar.
RBF ağı, doğrusal olmayan karmaşık ilişkileri modellemek için kullanılan güçlü bir yöntemdir.

Çalışma Prensibi:
1. Veri uzayına Gaussian fonksiyonları yerleştirilir (RBF merkezleri)
2. Her RBF'in genişliği (sigma) ve ağırlığı (x) belirlenir
3. Tüm RBF'lerin ağırlıklı toplamı ile tahmin yapılır

Avantajları:
- Doğrusal olmayan ilişkileri modelleyebilir
- Yerel özellikleri yakalayabilir
- Eğitimi hızlıdır
"""

import numpy as np
import math
from dataset3 import ti,yi
import matplotlib.pyplot as plt


def gaussianFunction(t,c,sigma):
    """
    Gaussian (RBF) fonksiyonu
    
    Parametreler:
    t: Giriş değeri
    c: RBF merkezi
    sigma: RBF genişliği (yayılım parametresi)
    """
    h = math.exp(-(t-c)**2/(sigma**2))
    return h

def RBFIO(t,x,c,sigma):
    """
    RBF ağının çıkışını hesaplar
    
    Parametreler:
    t: Giriş değerleri
    x: RBF ağırlıkları
    c: RBF merkezleri
    sigma: RBF genişlikleri
    """
    ythat = []
    for ti in t: 
        toplam = 0
        for i in range(0,len(x)):
            # Her RBF'in ağırlıklı katkısını topla
            toplam += x[i]*gaussianFunction(ti,c[i],sigma[i])
        ythat.append(toplam)
    return ythat

def findXCS(ti,yi,RBF_count):
    """
    RBF parametrelerini belirler
    
    Parametreler:
    ti: Giriş verileri
    yi: Çıkış verileri
    RBF_count: Kullanılacak RBF sayısı
    
    Dönüş:
    x: RBF ağırlıkları
    c: RBF merkezleri
    s: RBF genişlikleri
    """
    # Veri aralığını RBF sayısına göre böl
    lengthOfSegment = (max(ti)-min(ti))/RBF_count
    
    # Her RBF için aynı genişliği kullan
    s = [lengthOfSegment for tmp in range(0,RBF_count)]
    
    # RBF merkezlerini eşit aralıklarla yerleştir
    c = [min(ti)+ (lengthOfSegment/2) + (lengthOfSegment*tmp) for tmp in range(0,RBF_count)]
    
    numOfData = len(ti)
    
    # Jacobian matrisini oluştur
    J = np.zeros((numOfData,RBF_count))
    for i in range(0,numOfData):
        for j in range(0,RBF_count):
            J[i][j] = -gaussianFunction(ti[i],c[j],s[j])

    # En küçük kareler yöntemi ile ağırlıkları bul
    A = np.linalg.inv(J.transpose().dot(J))
    B = J.transpose().dot(yi)
    x = -A.dot(B)

    return x,c,s

def plotResult(ti,yi,x,c,s,fvalidation):
    """
    RBF modelinin sonuçlarını görselleştirir
    """
    T = np.arange(min(ti),max(ti),0.1)
    ythat = RBFIO(T,x,c,s)
    plt.scatter(ti,yi,label='Training Data',color='blue',marker='x')
    plt.plot(T,ythat,label='RBF Model',color='red',linestyle='solid',linewidth=1)
    plt.xlabel('ti')
    plt.ylabel('yi')
    plt.title(str(len(x)) + ' düğümlü RBF Modeli | FV: ' + str(fvalidation),fontstyle='italic')
    plt.grid(color='gray',linestyle='--',linewidth=0.1)
    plt.legend()
    plt.show()

# Veriyi eğitim ve doğrulama setlerine ayır
trainingIndices = np.arange(0,len(ti),2)      # Çift indeksli veriler eğitim için
trainingInput = np.array(ti)[trainingIndices]
trainingOutput = np.array(yi)[trainingIndices]

validationIndices = np.arange(1,len(ti),2)    # Tek indeksli veriler doğrulama için
validationInput = np.array(ti)[validationIndices]
validationOutput = np.array(yi)[validationIndices]

# Farklı sayıda RBF kullanarak modeller oluştur ve karşılaştır
RBF = []; FV = []
for RBF_count in range(1,10):
    # RBF parametrelerini bul
    x,c,s = findXCS(trainingInput,trainingOutput,RBF_count)
    
    # Doğrulama seti üzerinde performansı hesapla
    yhat = RBFIO(validationInput,x,c,s)
    e = np.array(validationOutput)-np.array(yhat)
    fvalidation = sum(e**2)
    
    # Sonuçları kaydet
    RBF.append(RBF_count)
    FV.append(fvalidation)
    print(f"RBF sayısı: {RBF_count}, Doğrulama hatası: {fvalidation}")
    plotResult(ti,yi,x,c,s,fvalidation)

# Farklı RBF sayılarının performansını karşılaştır
plt.bar(RBF,FV,color='darkred')
plt.xlabel('RBF Sayısı')
plt.ylabel('Doğrulama Performansı')
plt.title('RBF Modeli',fontstyle='italic')
plt.grid(color='gray',linestyle='--',linewidth=0.1)
plt.show()

