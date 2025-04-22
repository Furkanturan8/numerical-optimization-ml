# Polinomik Regresyon Analizi

## Genel Bakış
Polinomik regresyon, veri noktaları arasındaki ilişkiyi modellemek için kullanılan bir regresyon analizi yöntemidir. Bu yöntem, doğrusal regresyonun daha genel bir halidir ve veri noktaları arasındaki doğrusal olmayan ilişkileri modellemek için kullanılır.

## Temel Kavramlar

### Polinomik Regresyon Nedir?
- **Yöntem mi?** Evet, polinomik regresyon bir analiz ve modelleme yöntemidir.
- **Model mi?** Evet, aynı zamanda bir matematiksel modeldir. Bu yöntem sonucunda elde edilen polinom denklemi bir modeldir.

### Matematiksel İfade
Polinomik regresyon modeli şu şekilde ifade edilir:

y = β₀ + β₁x + β₂x² + β₃x³ + ... + βₙxⁿ + ε

Burada:
- y: Bağımlı değişken
- x: Bağımsız değişken
- β₀, β₁, β₂, ..., βₙ: Polinom katsayıları
- n: Polinomun derecesi
- ε: Hata terimi

## Bu Projede Kullanılan Yöntemler

1. **En Küçük Kareler Yöntemi (Least Squares)**
   - Polinom katsayılarını bulmak için kullanılır
   - Hata karelerinin toplamını minimize eder

2. **K-fold Benzeri Validation**
   - Modelin genelleme yeteneğini test etmek için kullanılır
   - Veri seti eğitim ve doğrulama olarak bölünür

3. **Model Seçimi**
   - Farklı derece polinomlar denenir
   - En iyi performansı gösteren model seçilir

## Kullanım Alanları

Polinomik regresyon şu alanlarda sıklıkla kullanılır:
- Bilimsel veri analizi
- Mühendislik uygulamaları
- Ekonomik tahminler
- Zaman serisi analizi
- Doğrusal olmayan ilişkilerin modellenmesi

## Avantajları ve Dezavantajları

### Avantajlar
- Doğrusal olmayan ilişkileri modelleyebilme
- Kolay yorumlanabilir sonuçlar
- Esnek model yapısı

### Dezavantajlar
- Aşırı uyum (overfitting) riski
- Yüksek dereceli polinomlarda hesaplama karmaşıklığı
- Extrapolation'da (tahmin aralığı dışında) güvenilirlik azalması

## Örnek Uygulama
Bu projede, verilen zaman serisi verilerine en uygun polinom modelini bulmak için polinomik regresyon analizi kullanılmıştır. Program, farklı derece polinomları deneyerek en iyi modeli seçer ve sonuçları görselleştirir. 