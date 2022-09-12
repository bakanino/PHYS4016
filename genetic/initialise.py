import numpy as np

population_size = 10

np.random.seed(0)
P = 1e8 * np.random.rand(population_size, 1)
K = 0.2 * np.random.rand(population_size, 1)
phi = 2 * np.pi * np.random.rand(population_size, 1)
fitness = np.zeros((population_size, 1))

population = np.hstack((P, K, phi, fitness))
print(population)

