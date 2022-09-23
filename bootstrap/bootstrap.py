import numpy as np
import matplotlib.pyplot as plt

lsat = np.array([576, 635, 558, 578, 666, 580, 555, 661, 651, 605, 653, 575, 545, 572, 594])
gpa = np.array([3.3900, 3.3000, 2.8100, 3.0300, 3.4400, 3.0700, 3.0000, 3.4300, 3.3600, 3.1300, 3.1200, 2.7400, 2.7600, 2.8800, 2.9600])

N = len(lsat)
N_expts = 10000

np.random.seed(0)

rhos = []
for i in range(N_expts):
    samples = np.random.randint(N, size=N)
    lsat_samples = lsat[samples]
    gpa_samples = gpa[samples]
    rho_samples = np.corrcoef(lsat_samples, gpa_samples)[0][1]
    rhos.append(rho_samples)

plt.hist(rhos, bins=50, edgecolor='black')
plt.xlabel(r'$\rho$')
plt.ylabel('Counts')
plt.tight_layout()
plt.savefig('rho_distribution.png')
