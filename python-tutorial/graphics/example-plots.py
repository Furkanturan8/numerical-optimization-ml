"""
Matplotlib Grafik Örnekleri
"""
import numpy as np
import matplotlib.pyplot as plt

def line_plot_example():
    """Basit çizgi grafiği örneği"""
    x = np.linspace(-5, 5, 100)
    y1 = x**2
    y2 = x**3
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y1, 'r-', label='x²')
    plt.plot(x, y2, 'b--', label='x³')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Çizgi Grafiği Örneği')
    plt.grid(True)
    plt.legend()
    plt.show()

def contour_plot_example():
    """Kontur grafiği örneği"""
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2  # f(x,y) = x² + y²
    
    plt.figure(figsize=(10, 8))
    contour = plt.contour(X, Y, Z, levels=20)
    plt.clabel(contour, inline=True, fontsize=8)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Kontur Grafiği Örneği: f(x,y) = x² + y²')
    plt.colorbar(label='f(x,y) değeri')
    plt.grid(True)
    plt.show()

def surface_plot_example():
    """3D yüzey grafiği örneği"""
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2  # f(x,y) = x² + y²
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('3D Yüzey Grafiği Örneği: f(x,y) = x² + y²')
    fig.colorbar(surf, label='f(x,y) değeri')
    plt.show()

if __name__ == "__main__":
    # Tüm örnekleri çalıştır
    print("Çizgi grafiği gösteriliyor...")
    line_plot_example()
    
    print("\nKontur grafiği gösteriliyor...")
    contour_plot_example()
    
    print("\n3D yüzey grafiği gösteriliyor...")
    surface_plot_example() 