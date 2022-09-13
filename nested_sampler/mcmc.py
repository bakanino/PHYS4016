import numpy as np


def likelihood(x, y):
    return -1/5428 * (np.sin(x) * np.exp((1 - np.cos(y))**2) + np.cos(y) *
                      np.exp((1 - np.sin(x))**2) + (x - y)**2 - 97.8)


def mcmc(x, y, likelihood_rejected, low_x, high_x, low_y, high_y):
    accepted = 0
    while accepted < 100:
        proposed_x = x + 0.13 * np.random.randn()
        proposed_y = y + 0.2 * np.random.randn()

        if (proposed_x < low_x or proposed_x > high_x or proposed_y < low_y or proposed_y > high_y):
            continue

        proposed_likelihood = likelihood(proposed_x, proposed_y)

        if proposed_likelihood > likelihood_rejected:
            x = proposed_x
            y = proposed_y
            accepted += 1

    return x, y


np.random.seed(0)

# Specify bounds
low_x = -6.5
high_x = 0
low_y = -10
high_y = 0

x, y = mcmc(-3, -3, 0.02, low_x, high_x, low_y, high_y)

print('({:.4f}, {:.4f})'.format(x, y))
