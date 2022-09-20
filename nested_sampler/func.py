import numpy as np

def likelihood(x, y):
    return -1/5428 * (np.sin(x) * np.exp((1 - np.cos(y))**2) + np.cos(y) * \
    np.exp((1 - np.sin(x))**2) + (x - y)**2 - 97.8)

print(likelihood(-6, -4))
print(likelihood(-2, -3))
