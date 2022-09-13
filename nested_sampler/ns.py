import numpy as np
from funcs import *

np.random.seed(0)

# Constants
N = 500
low_x = -6.5
high_x = 0
low_y = -10
high_y = 0

# Generate active samples
active = np.zeros((N, 3))
active[:, 0] = np.random.uniform(low = low_x, high = high_x, size = N)
active[:, 1] = np.random.uniform(low = low_y, high = high_y, size = N)

for i in range(N):
    active[i, 2] = likelihood(active[i, 0], active[i, 1])

iteration = 0
while True:
    # Eliminate
    active, eliminated = eliminate(active)
    if iteration == 0:
        rejected = eliminated
    else:
        rejected = np.vstack((rejected, eliminated))

    # Pick a new point
    active = add_point(active, eliminated, low_x, high_x, low_y, high_y)

    # Check termination condition
    X = np.exp(-iteration/N)
    Lmax = active[:, -1].max()
    if np.abs(Lmax * X) < 1e-4:
        break

    iteration += 1

best = active[:, -1].argmax()
print('({:.1f}, {:.1f})'.format(active[best, 0], active[best, 1]))
