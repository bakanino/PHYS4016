import numpy as np


def eliminate(active):
    worst = active[:, -1].argmin()
    eliminated = active[worst]
    active = np.delete(active, worst, 0)
    return active, eliminated


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


def add_point(active, eliminated, low_x, high_x, low_y, high_y):
    threshold = eliminated[-1]
    while True:
        ind = np.random.randint(low=0, high=active.shape[0])
        proposed_x, proposed_y = mcmc(active[ind, 0], active[ind, 1], threshold, low_x, high_x, low_y, high_y)
        proposed_likelihood = likelihood(proposed_x, proposed_y)
        if proposed_likelihood > threshold:
            break
    active = np.vstack((active, [[proposed_x, proposed_y, proposed_likelihood]]))
    return active


np.random.seed(0)

# Specify bounds
low_x = -6.5
high_x = 0
low_y = -10
high_y = 0

#x, y = mcmc(-3, -3, 0.02, low_x, high_x, low_y, high_y)
#
#print('({:.4f}, {:.4f})'.format(x, y))

active = np.array([[-6, -4, 0.0167],
                   [-2, -3, 0.0336],
                   [ 0,  0, 0.0175],
                   [-3, -7, 0.0146]])

eliminated = [-8, -1, 0.0103]

active = add_point(active, eliminated, low_x, high_x, low_y, high_y)

#print(active)
