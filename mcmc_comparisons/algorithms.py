import numpy as np
import matplotlib.pyplot as plt

class Race():
    def __init__(self, function, bounds, start=np.array([0., 0.])):
        self.function = function
        self.bounds = bounds
        self.n_chain = int(1e4)
        self.start = start
        self.MCMC_chain = []
        self.HMC_chain = []
        self.GW_MCMC_chain = []

    def race(self):
        self.MCMC()
        print('Finished MCMC')
        self.GW_MCMC()
        print('Finished affine invariant MCMC')
        self.HMC()
        print('Finished HMC')
        self.plot()
        pass

    def MCMC(self):
        # Sigmas for proposal
        sigma_x = sigma_y = 1

        # Starting point
        if self.MCMC_chain == []:
            x_current = self.start[0]
            y_current = self.start[1]
        else:
            x_current = self.MCMC_chain[-1, 0]
            y_current = self.MCMC_chain[-1, 1]


        # Log likelihood because a gaussian error would be proportional to exp(loglike)
        log_like = self.function(np.array([x_current, y_current]))

        # for i in range(len(self.MCMC_chain), self.n_chain):
        for i in range(self.n_chain):
            # Select from proposal distribution
            proposed_x = x_current + sigma_x * np.random.normal()
            proposed_y = y_current + sigma_y * np.random.normal()

            # Check bounds
            if not (self.bounds[0] <= proposed_x <= self.bounds[1]) or not (self.bounds[2] <= proposed_y <= self.bounds[3]):
                self.MCMC_chain.append(np.array([x_current, y_current]))
                continue

            # Find misfit
            proposed_log_like = self.function(np.array([proposed_x, proposed_y]))

            # Compare proposed location with present (Metropolis Ratio)
            rel_prob = proposed_log_like / log_like

            # Decide whether to to take a step based on a random number
            random_value = np.random.uniform()
            if random_value < rel_prob:
                x_current = proposed_x
                y_current = proposed_y
                log_like = proposed_log_like

            self.MCMC_chain.append(np.array([x_current, y_current]))

    def HMC(self):
        def leapfrog_integrator(x):
            p = np.random.randn(2)
            p0 = p.copy()
            T = 37
            eps = 0.1
            for n in range(int(T/eps)):
                p = p - eps/2 * self.function(x, gradient=True)
                x = x + eps*p
                p = p - eps/2 * self.function(x, gradient=True)

            p = p.reshape(2, 1)
            p0 = p0.reshape(2, 1)
            return x, p, p0

        x0 = self.start.copy()
        for i in range(self.n_chain):
            x, p, p0 = leapfrog_integrator(x0.copy())

            H = -np.log(self.function(x)) + 0.5*p.T@p
            H0 = -np.log(self.function(x0)) + 0.5*p0.T@p0
            alpha = np.exp(-H)/np.exp(-H0)
            if np.random.uniform() <= alpha and (self.bounds[0] <= x[0] <= self.bounds[1]) and (self.bounds[2] <= x[1] <= self.bounds[3]):
                x0 = x.copy()
            self.HMC_chain.append(x0.copy())

    def GW_MCMC(self):

        def distribution(z):
            a = 2
            if z >= 1/a and z <= a:
                return 1/np.sqrt(z)
            else:
                return 0

        def sample_from_distribution():
            while True:
                x = np.random.uniform(high=2)
                y = np.random.uniform()
                if distribution(x) >= y:
                    return x

        num_walkers = 250
        old_walkers = np.random.normal(scale=0.5, size=(num_walkers, 2))
        new_walkers = old_walkers.copy()

        for iteration in range(self.n_chain//num_walkers):
            for k in range(num_walkers):
                while True:
                    j = np.random.choice(range(num_walkers))
                    if j != k:
                        break
                j_walker = old_walkers[j]

                z = sample_from_distribution()
                Y = j_walker + z*(old_walkers[k] - j_walker)
                q = z * self.function(Y)/self.function(old_walkers[k])
                r = np.random.uniform()
                if r <= q and (self.bounds[0] <= Y[0] <= self.bounds[1]) and (self.bounds[2] <= Y[1] <= self.bounds[3]):
                    new_walkers[k] = Y
                else:
                    new_walkers[k] = old_walkers[k].copy()

            self.GW_MCMC_chain.append(new_walkers.copy())
            old_walkers = new_walkers.copy()


    def plot(self):
        # Generate heatmap
        burn_in = 0
        bins = 50

        x = np.linspace(self.bounds[0], self.bounds[1], 50)
        y = np.linspace(self.bounds[2], self.bounds[3], 50)
        xx, yy = np.meshgrid(x, y)
        zz = np.zeros(xx.shape)
        for i in range(zz.shape[0]):
            for j in range(zz.shape[1]):
                zz[i, j] = self.function(np.array([xx[i, j], yy[i, j]]))

        zz /= zz.sum()

        plt.figure(0)
        z_vals = zz.sum(axis=1)
        z_vals = z_vals/z_vals.mean() * (1/(self.bounds[1] - self.bounds[2]))
        plt.plot(y, z_vals, c='black')

        plt.figure(1)
        z_vals = zz.sum(axis=0)
        z_vals = z_vals/z_vals.mean() * (1/(self.bounds[1] - self.bounds[2]))
        plt.plot(x, z_vals, c='black')

        vmax, vmin = 0, 0
        for i, list_chain in enumerate([self.MCMC_chain, self.GW_MCMC_chain, self.HMC_chain]):
            chain = np.vstack(list_chain)
            x = chain[burn_in:, 0]
            y = chain[burn_in:, 1]
            heatmap, _, _ = np.histogram2d(x, y, range=[self.bounds[:2], self.bounds[2:]], bins=bins, density=True)
            vmax = max(heatmap.max(), vmax)

        for i, list_chain in enumerate([self.MCMC_chain, self.GW_MCMC_chain, self.HMC_chain]):
            chain = np.vstack(list_chain)
            x = chain[burn_in:, 0]
            y = chain[burn_in:, 1]


            heatmap, x_edges, y_edges = np.histogram2d(x, y, range=[self.bounds[:2], self.bounds[2:]], bins=bins, density=True)
            extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
            x_vals = [(x_edges[i] + x_edges[i+1])/ 2 for i in range(bins)]
            y_vals = [(y_edges[i] + y_edges[i+1])/ 2 for i in range(bins)]

            # MPP
            plt.figure(0)
            z_vals = heatmap.sum(axis = 0)
            z_vals = z_vals/z_vals.mean() * (1/(self.bounds[1] - self.bounds[2]))
            plt.plot(y_vals, z_vals)
            plt.xlabel('$x_1$')
            plt.ylabel('Marginal Posterior Probability')
            plt.title('Marginal Posterior Probability of $x_1$')
            plt.xlim(self.bounds[2:])
            plt.legend(['True', 'MCMC', 'Affine invariant MCMC', 'HMC'])
            plt.savefig('mpp_omega.png')

            plt.figure(1)
            z_vals = heatmap.sum(axis = 1)
            z_vals = z_vals/z_vals.mean() * (1/(self.bounds[1] - self.bounds[2]))
            plt.plot(x_vals, z_vals)
            plt.xlabel('$x_0$')
            plt.ylabel('Marginal Posterior Probability')
            plt.title('Marginal Posterior Probability of $x_0$')
            plt.xlim(self.bounds[:2])
            plt.legend(['True', 'MCMC', 'Affine invariant MCMC', 'HMC'])
            plt.savefig('mpp_c.png')

            plt.figure()
            plt.imshow(np.rot90(heatmap), extent = extent, aspect = 'equal', vmin=0, vmax=vmax)
            plt.xlabel('$x_0$')
            plt.ylabel('$x_1$')
            if i == 0:
                MCMC_type = 'MCMC'
            elif i == 1:
                MCMC_type = 'AI MCMC'
            elif i == 2:
                MCMC_type = 'HMC'
            plt.title(f'Histogram Density Image for {MCMC_type}')
            plt.xlim(self.bounds[:2])
            plt.ylim(self.bounds[2:])
            plt.colorbar()
            plt.savefig(f'density_image_{i}.png')

