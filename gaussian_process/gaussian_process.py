import george
from george.kernels import ExpSquaredKernel, ExpSine2Kernel, RationalQuadraticKernel, ExpSquaredKernel
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

npz_file = np.load('mauna_loa.npz')
date_train = npz_file['date_train']
date_test = npz_file['date_test']
co2_train = npz_file['co2_train']
co2_test = npz_file['co2_test']

def neg_log_likelihood(parameters):
    gp.set_parameter_vector(parameters)
    log_like = gp.log_likelihood(co2_train, quiet=True)
    return -log_like if np.isfinite(log_like) else 1e25

def grad_neg_log_likelihood(parameters):
    gp.set_parameter_vector(parameters)
    return -gp.grad_log_likelihood(co2_train, quiet=True)

long_term_kernel = 10**2 * ExpSquaredKernel(metric=100)
periodic_kernel = 5**2 * ExpSquaredKernel(metric=1) * ExpSine2Kernel(gamma=1, log_period=0)
medium_term_kernel = RationalQuadraticKernel(log_alpha=0, metric=1)
kernel = long_term_kernel + periodic_kernel + medium_term_kernel
gp = george.GP(kernel, mean=np.mean(co2_train), fit_mean=True, white_noise=1, fit_white_noise=True)
gp.compute(date_train)
x0 = gp.get_parameter_vector()
results = minimize(neg_log_likelihood, x0, jac=grad_neg_log_likelihood)
#print(results.x)
gp.set_parameter_vector(results.x)
# predict for 2000 evenly spaced points between [2010, 2030]
x_test = np.linspace(2010, 2030, 2000)
mu, var = gp.predict(co2_train, x_test, return_var=True)
stdev = np.sqrt(var)
plt.plot(x_test, mu, label='$\mu$')
plt.plot(date_test, co2_test, 'ko', markersize=2, label='Measurements')
plt.fill_between(x_test, mu-stdev, mu+stdev, color='tab:green', alpha=0.2, label='$2\sigma$')
plt.xlabel('Date')
plt.ylabel('CO$_2$ (ppm)')
plt.title('GP prediction of CO$_2$ concentrations')
plt.xlim([2010, 2030])
plt.legend()
plt.savefig('gp.png')
