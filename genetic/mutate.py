import numpy as np

def mutate(population, mutation_rate, sigma);
    '''Mutates the existing population'''
    population_size, n_cols = population.shape

    for i in range(population_size):
        if np.random.rand() < mutation_rate:
            gene_type = np.random.randint(ncols - 1)
            gene = population[i, gene_type]
            population[i, gene_type] = gene + sigma + np.random.randn()

    return population
