import numpy as np
np.seterr(over='ignore')
import matplotlib.pyplot as plt

npz_file = np.load('data.npz')
data = npz_file['data']
z = npz_file['z']
noise = npz_file['noise']

# Set seed
np.random.seed(0)

# Markov chain length
n_chain = 100

# Starting point
C_current = 5.0
Om_current = 0.5
proposed = C_current / (np.sqrt(Om_current*z + 1) - 1) + 200*np.sinh(Om_current*z)/np.sqrt(C_current)

# Sigma for proposal
sigma_C = 0.25
sigma_Om = 0.025

log_like = np.sum(-(data - proposed)**2 / (2*noise**2))
chain_history = np.zeros((n_chain, 2))

for i in range(n_chain):
    # Select from proposal distribution
    proposed_C = C_current + sigma_C * np.random.randn()
    proposed_Om = Om_current + sigma_Om * np.random.randn()

    # Check bounds
    if (proposed_C < 1) or (proposed_C > 5) or (proposed_Om < 0.1 ) or (proposed_Om > 1):
        chain_history[i, 0] = C_current
        chain_history[i, 1] = Om_current
        continue

    # Predict values using this model
    proposed = proposed_C / (np.sqrt(proposed_Om*z + 1) - 1) + 200*np.sinh(proposed_Om*z)/np.sqrt(proposed_C)
    proposed_log_like = np.sum(-(data - proposed)**2 / (2*noise**2))

    # Compare proposed location with present (Metropolis Ratio)
    rel_prob = np.exp(proposed_log_like - log_like)

    # Decide whether to take a step based on a random number
    random_value = np.random.rand()
    if random_value < rel_prob:
        C_current = proposed_C
        Om_current = proposed_Om
        log_like = proposed_log_like

    chain_history[i, 0] = C_current
    chain_history[i, 1] = Om_current

#print(chain_history[:100])
plt.scatter(chain_history[:, 0], chain_history[:, 1], s=1, c=range(chain_history.shape[0]), cmap='viridis')
plt.xlabel('$C$')
plt.ylabel('$\Omega_m$')
plt.title('MCMC Walk')
plt.colorbar().ax.set_ylabel('Iteration')
plt.savefig('plot')






