import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt
np.random.seed(0)

def exp_quadratic(xa, xb):
    return np.exp(-scipy.spatial.distance.cdist(xa, xb, 'minkowski', p=2)**2 / 2)

N_FUNCS = 5
x = np.linspace(-3, 3, 50).reshape(-1, 1)
z = exp_quadratic(x, x)
means = np.zeros(50)

plt.imshow(z, extent=[-3, 3, -3, 3])
plt.title('Exponentiated quadratic\nexample of a covariance matrix')
plt.xlabel('$x_a$')
plt.ylabel('$x_b$')
plt.colorbar()
plt.savefig('plot.png')

funcs = np.random.multivariate_normal(means, z, size=N_FUNCS)
#for k in range(N_FUNCS):
#    plt.plot(x, funcs[k,:], '.-')
plt.plot(x, funcs.T, '.-')
