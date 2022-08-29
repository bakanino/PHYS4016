import numpy as np
import matplotlib.pyplot as plt
np.seterr(divide='ignore', over='ignore')

npz_file = np.load('data.npz')
data = npz_file['data']
z = npz_file['z']
noise = npz_file['noise']

# Set seed
np.random.seed(0)

# Markov chain length
n_chain = int(5e5)

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

    # Find misfit
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

# Generate heatmap
burn_in = 200
x = chain_history[burn_in:, 0]
y = chain_history[burn_in:, 1]

bins = 50
heatmap, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
#print(np.isfinite(heatmap).all())
heatmap /= len(x)
extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
x_vals = [(x_edges[i] + x_edges[i+1]) / 2 for i in range(bins)]
y_vals = [(y_edges[i] + y_edges[i+1]) / 2 for i in range(bins)]

plt.figure
plt.imshow(np.rot90(heatmap), extent=extent, aspect='auto')
plt.xlabel('$C$')
plt.ylabel('$\Omega_m$')
plt.title('Histogram Density Image')
plt.savefig('density_image.png')
