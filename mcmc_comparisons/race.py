import numpy as np
from algorithms import Race

def sphere(x, gradient=False):
    if gradient:
        return 2*x
    return np.exp(-np.sum(x**2))

def warped_sphere(x, gradient=False):
    a = 10
    if gradient:
        grad_values = np.zeros(2)
        grad_values[0] = 2*x[0]
        grad_values[1] = 2*a*x[1]
        return grad_values
    return np.exp(-(x[0]**2 + a*x[1]**2))

def warped_gaussian(x, gradient=False):
    eps = 0.01
    if gradient:
        grad_values = np.zeros(2)
        grad_values[0] = (x[0] - x[1])/eps + (x[0] + x[1])
        grad_values[1] = -(x[0] - x[1])/eps + (x[0] + x[1])
        return grad_values
    return np.exp(-(x[0]-x[1])**2 / (2*eps) - (x[0]+x[1])**2 / 2)

def rosenbrocks_banana(x, gradient=False):
    a = 1
    b = 100
    if gradient:
        grad_values = np.zeros(2)
        grad_values[0] = -4*x[0]*b * (x[1] - x[0]**2) - 2*(a - x[0])
        grad_values[1] = 2*b*(x[1] - x[0]**2)
        return grad_values/20
    return np.exp(-(b*(x[1]-x[0]**2)**2 + (1-x[0])**2)/20)

bounds = [-2, 2, -2, 2]
start = np.array([1., 1.])
race = Race(sphere, bounds, start=start)
race.race()
