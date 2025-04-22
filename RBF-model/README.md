# Radyal Tabanlı Fonksiyon (RBF) Modeli

## RBF Nedir?
RBF (Radyal Tabanlı Fonksiyon), bir noktanın merkeze olan uzaklığına bağlı olarak değer üreten özel bir fonksiyon türüdür. En yaygın kullanılan RBF, Gaussian (çan eğrisi) fonksiyonudur.

## Model mi, Yöntem mi?
RBF **hem bir model hem de bir yöntemdir**:
- **Model olarak**: Veri noktaları arasındaki ilişkiyi temsil eden matematiksel bir yapıdır.
- **Yöntem olarak**: Veri modelleme ve fonksiyon yaklaşımı için kullanılan bir tekniktir.

## RBF'in Temel Bileşenleri

### 1. Merkez Noktaları (c)
- Veri uzayında stratejik olarak yerleştirilen referans noktalarıdır
- Her RBF fonksiyonu bir merkez noktasına sahiptir

### 2. Genişlik Parametresi (σ - sigma)
- RBF'in etki alanını belirler
- Fonksiyonun ne kadar "geniş" veya "dar" olacağını kontrol eder

### 3. Ağırlıklar (w)
- Her RBF'in çıktıya olan katkısını belirler
- Eğitim sırasında optimize edilir

## Matematiksel İfade

RBF ağının çıktısı şu şekilde hesaplanır:

y(x) = Σ wᵢ × exp(-(‖x-cᵢ‖²)/(2σᵢ²))

Burada:
- x: Giriş değeri
- cᵢ: i. RBF'in merkezi
- σᵢ: i. RBF'in genişlik parametresi
- wᵢ: i. RBF'in ağırlığı

## Yapay Sinir Ağlarındaki Yeri

RBF ağları, özel bir yapay sinir ağı türüdür:
- Genellikle 3 katmanlı bir yapıya sahiptir:
  1. Giriş katmanı
  2. RBF katmanı (gizli katman)
  3. Çıkış katmanı
- Klasik yapay sinir ağlarından farklı olarak:
  - Sadece tek gizli katman kullanır
  - Aktivasyon fonksiyonu olarak RBF kullanır
  - Eğitimi genellikle daha hızlıdır

## Kullanım Alanları

1. **Fonksiyon Yaklaşımı**
   - Karmaşık matematiksel fonksiyonların modellenmesi
   - Zaman serisi tahminleri

2. **Örüntü Tanıma**
   - Görüntü işleme
   - Ses tanıma
   - El yazısı tanıma

3. **Kontrol Sistemleri**
   - Robot kontrolü
   - Proses kontrolü

4. **Finansal Tahmin**
   - Borsa tahminleri
   - Risk analizi

5. **Sinyal İşleme**
   - Gürültü giderme
   - Sinyal sınıflandırma

## Avantajları

1. **Hızlı Eğitim**
   - Klasik sinir ağlarına göre daha hızlı eğitilir
   - Yerel minimum sorununa daha az duyarlıdır

2. **Yorumlanabilirlik**
   - Her RBF'in etki alanı görselleştirilebilir
   - Model davranışı daha kolay anlaşılır

3. **Esneklik**
   - Doğrusal olmayan problemleri modelleyebilir
   - Farklı RBF türleri kullanılabilir

## Dezavantajları

1. **Merkez Seçimi**
   - Merkez noktalarının belirlenmesi kritiktir
   - Yanlış merkez seçimi performansı düşürür

2. **Boyut Sorunu**
   - Yüksek boyutlu problemlerde RBF sayısı çok artabilir
   - Hesaplama maliyeti yükselebilir

3. **Genişlik Parametresi**
   - Optimal genişlik parametresinin belirlenmesi zor olabilir
   - Model performansını önemli ölçüde etkiler

## Bu Projede Kullanım

Bu projede RBF modeli:
- Veri noktalarına en iyi uyan modeli bulmak için kullanılmıştır
- Farklı sayıda RBF merkezi denenerek optimal model aranmıştır
- En küçük kareler yöntemi ile ağırlıklar belirlenmiştir
- Çapraz doğrulama ile model performansı değerlendirilmiştir 