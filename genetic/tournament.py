import numpy as np


def tournament(population, selection_pressure):
    population_size = population.shape[0]
    N_tournaments = 2*population_size

    winners = []

    for i in range(N_tournaments):
        contestant_1 = np.random.randint(population_size)
        contestant_2 = np.random.randint(population_size)

        fitness_1 = population[contestant_1, 3]
        fitness_2 = population[contestant_2, 3]

    return winners

np.random.seed(0)

population = np.array([[1e7,    1, 54.5633],
                       [1e6, 0.01, 0.66232],
                       [5e7,    2, 209.129]])

selection_pressure = 0.85

gene_pool = tournament(population, selection_pressure)
print(gene_pool)
