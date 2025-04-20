import numpy as np

# Rastgele sayı üretimi için seed belirleme (tekrar üretilebilirlik için)
np.random.seed(42)

# Zaman değerlerini oluşturma (0'dan 10'a kadar 20 nokta)
ti = np.linspace(0, 10, 20)

# Gerçek bir fonksiyon oluşturma (y = 2x^2 - 3x + 1 + gürültü)
yi = 2 * ti**2 - 3 * ti + 1 + np.random.normal(0, 5, ti.shape)

# NumPy dizilerini liste formatına çevirme
ti = ti.tolist()
yi = yi.tolist() 