# Ev Fiyat Tahmini Örneği

Bu örnek, denetimli öğrenmenin regresyon problemi için hazırlanmış bir çalışmadır. Amacımız, evin çeşitli özelliklerini kullanarak fiyatını tahmin eden bir model geliştirmektir.

## Veri Seti Hakkında

Veri seti `house_prices.csv` dosyasında bulunmaktadır ve aşağıdaki özellikleri içermektedir:

1. **metrekare**: Evin büyüklüğü (m²)
2. **yatak_odasi**: Yatak odası sayısı
3. **banyo_sayisi**: Banyo sayısı
4. **ev_yasi**: Evin yaşı (yıl)
5. **merkeze_uzaklik**: Şehir merkezine uzaklık (km)
6. **fiyat**: Ev fiyatı (TL) - Hedef değişken

## Veri Seti Özellikleri

- Toplam 20 örnek içermektedir (son sürümde 170 e çıkarıldı.)
- Tüm değerler gerçekçi aralıklarda oluşturulmuştur
- Fiyatlar 350,000 TL ile 1,900,000 TL arasında değişmektedir
- Veriler arasında eksik değer bulunmamaktadır

## Model Geliştirme Adımları

1. **Veri Ön İşleme**

   - Verilerin yüklenmesi
   - Özelliklerin normalize edilmesi
   - Eğitim ve test setlerine ayırma

2. **Model Eğitimi**

   - Doğrusal Regresyon modelinin oluşturulması
   - Gradient Descent ile model parametrelerinin optimizasyonu
   - Model performansının değerlendirilmesi

3. **Model Değerlendirme Metrikleri**
   - MAE (Mean Absolute Error)
   - MSE (Mean Squared Error)
   - R² Skoru

## Beklenen Çıktılar

Model eğitimi sonrasında:

- Test seti üzerinde R² > 0.8
- Makul MAE değerleri (örneğin < 100,000 TL)
- Gerçekçi tahminler

## Öneriler

- Farklı öğrenme oranları denenebilir
- Feature scaling yöntemleri karşılaştırılabilir
- Farklı regresyon modelleri ile sonuçlar kıyaslanabilir
