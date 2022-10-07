import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt

def exp_quadratic(xa, xb):
    return np.exp(-scipy.spatial.distance.cdist(xa, xb, 'minkowski', p=2)**2 / 2)

x = np.linspace(-3, 3, 50).reshape(-1, 1)
z = exp_quadratic(x, x)
plt.imshow(z, extent=[-3, 3, -3, 3])
plt.title('Exponentiated quadratic\nexample of a covariance matrix')
plt.xlabel('$x_a$')
plt.ylabel('$x_b$')
plt.colorbar()
plt.savefig('plot.png')
