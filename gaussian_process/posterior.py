import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.spatial
np.random.seed(0)

def exp_quadratic(xa, xb):
    return np.exp(-scipy.spatial.distance.cdist(xa, xb, 'minkowski', p=2)**2 / 2)

npz_file = np.load('data.npz')
x1 = npz_file['x1']
y1 = npz_file['y1']
x2 = np.linspace(-6, 6, 75).reshape(-1, 1)

sigma_11 = exp_quadratic(x1, x1)
sigma_12 = exp_quadratic(x1, x2)
sigma_22 = exp_quadratic(x2, x2)
sigma_11_I = np.linalg.inv(sigma_11)

mu_2 = (sigma_11_I @ sigma_12).T @ y1
sigma_2 = sigma_22 - (sigma_11_I @ sigma_12).T @ sigma_12
stdev = np.sqrt(np.diag(sigma_2))

print(mu_2)

plt.plot(x2, mu_2, label='$\mu_{2|1}$')
plt.plot(x1, y1, 'ko', label='$(x_1, y_1$')
plt.fill_between(x2.flat, mu_2-2*stdev, mu_2+2*stdev,
                 alpha=0.2, label='$2 \sigma_{2|1}$')
plt.legend()
plt.title('Distribution of posterior')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('pos_distribution.png')
