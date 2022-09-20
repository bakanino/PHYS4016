import numpy as np
import matplotlib.pyplot as plt

def likelihood(x, y):
    return -1/5428 * (np.sin(x) * np.exp((1 - np.cos(y))**2) + np.cos(y) * \
    np.exp((1 - np.sin(x))**2) + (x - y)**2 - 97.8)

x = np.linspace(-6.5, 0, 100)
y = np.linspace(-10, 0, 100)

xx, yy = np.meshgrid(x, y)
z = likelihood(xx, yy)

fig = plt.figure(figsize = (10, 7))
ax = fig.add_subplot(111, projection = '3d')
ax.plot_surface(xx, yy, z, cmap = 'rainbow')
ax.contour(xx, yy, z, zdir='z', offset=0, cmap='rainbow', levels = 50)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('likelihood')
plt.savefig('plot')
