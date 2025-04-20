import numpy as np

# Tekrar üretilebilirlik için seed belirleme
np.random.seed(42)

# Zaman değerlerini oluşturma (-5'ten 5'e kadar 30 nokta)
ti = np.linspace(-5, 5, 30)

# Doğrusal olmayan bir fonksiyon oluşturma
# f(x) = sin(x) + 0.5*cos(2x) + gürültü
yi = np.sin(ti) + 0.5 * np.cos(2*ti) + np.random.normal(0, 0.1, ti.shape)

# NumPy dizilerini liste formatına çevirme
ti = ti.tolist()
yi = yi.tolist() 