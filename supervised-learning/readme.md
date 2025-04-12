# Denetimli Öğrenme (Supervised Learning)

Denetimli öğrenme, makine öğrenmesinin temel yaklaşımlarından biridir. Bu yöntemde, modele etiketlenmiş veri setleri kullanılarak eğitim yaptırılır. Yani, her bir giriş verisinin karşılığında olması gereken çıkış değeri (etiket) önceden bilinmektedir.

## Denetimli Öğrenme Nasıl Çalışır?

1. **Veri Hazırlama**

   - Etiketlenmiş veri seti toplanır
   - Veriler eğitim ve test setlerine ayrılır
   - Veri ön işleme yapılır (normalizasyon, eksik verilerin doldurulması vb.)

2. **Model Seçimi**

   - Problemin türüne göre uygun model seçilir
   - Sınıflandırma için: Lojistik Regresyon, Karar Ağaçları, SVM
   - Regresyon için: Doğrusal Regresyon, Rastgele Orman, Gradyan Artırma

3. **Model Eğitimi**

   - Model, eğitim verileri kullanılarak eğitilir
   - Model, giriş verileri ile çıkış etiketleri arasındaki ilişkiyi öğrenir
   - Öğrenme sırasında model parametreleri optimize edilir

4. **Model Değerlendirmesi**
   - Eğitilen model test verileri üzerinde değerlendirilir
   - Performans metrikleri hesaplanır (doğruluk, hassasiyet, F1-skoru vb.)
   - Gerekirse model iyileştirmeleri yapılır

## Denetimli Öğrenme Türleri

### 1. Regresyon

Regresyon, sürekli sayısal değerlerin tahmin edilmesinde kullanılan bir denetimli öğrenme türüdür.

**Özellikler:**

- Çıktı değeri sürekli bir sayısal değerdir
- Tahmin edilen değer herhangi bir reel sayı olabilir
- Performans ölçümünde R², MSE, MAE gibi metrikler kullanılır

**Örnek Uygulamalar:**

- Ev/araba fiyat tahmini
- Sıcaklık tahmini
- Satış geliri tahmini
- Nüfus artış tahmini

### 2. Sınıflandırma

Sınıflandırma, verilerin önceden belirlenmiş kategorilere/sınıflara atanmasını sağlayan bir denetimli öğrenme türüdür.

**Özellikler:**

- Çıktı değeri kategoriktir (ayrık sınıflar)
- Tahmin sonucu belirli sınıflardan biridir
- Performans ölçümünde doğruluk, hassasiyet, F1-skoru gibi metrikler kullanılır

**Örnek Uygulamalar:**

- E-posta spam tespiti (spam/değil)
- Görüntü sınıflandırma (kedi/köpek/kuş)
- Hastalık teşhisi (hasta/sağlıklı)
- Kredi risk değerlendirmesi (riskli/risksiz)

## Kullanım Alanları

- Görüntü sınıflandırma
- Spam e-posta tespiti
- Hastalık teşhisi
- Fiyat tahmini
- Müşteri segmentasyonu

## Avantajları ve Dezavantajları

### Avantajları

- Yüksek doğruluk
- Kolay yorumlanabilirlik
- Özel durumları öğrenebilme

### Dezavantajları

- Çok miktarda etiketli veriye ihtiyaç duyar
- Etiketleme maliyetli olabilir
- Aşırı öğrenme riski
