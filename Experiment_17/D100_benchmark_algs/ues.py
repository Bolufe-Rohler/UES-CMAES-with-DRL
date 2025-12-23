import numpy as np
from functions import *


def row_norm(x):
    return x / np.sqrt(np.square(x).sum(axis=1))[:, np.newaxis]


def merge_pops(population, f_pop, pop_size, leaders, f_leaders, dim):
    indexes = np.argsort(f_pop)
    merged = np.zeros((2 * pop_size, dim))
    f_merged = np.zeros((2 * pop_size,))

    merged[0:pop_size, :] = population[indexes[0:pop_size], :]
    f_merged[0:pop_size] = f_pop[indexes[0:pop_size]]
    merged[pop_size:2 * pop_size] = leaders
    f_merged[pop_size:2 * pop_size] = f_leaders

    indexes = np.argsort(f_merged)
    leaders = merged[indexes[0:pop_size], :]
    f_leaders = f_merged[indexes[0:pop_size]]
    return leaders, f_leaders


def ues(fun, dim, max_eval, pop_size, bound):

    # MPS parameters
    alpha = 0.1
    gamma = 2
    d = np.sqrt(dim)*2*bound

    # Initial population
    population = np.zeros((2*pop_size, dim))
    f_pop = 1e+50 * np.ones((2*pop_size, ))

    leaders = np.multiply(bound, np.random.uniform(-1, 1, (pop_size, dim)))
    f_leaders = fun(leaders)
    count_eval = pop_size

    current_median = np.median(f_leaders)

    while count_eval < max_eval:
        new_median = np.median(f_pop)
        if current_median > new_median:
            current_median = new_median

            leaders, f_leaders = merge_pops(population, f_pop, pop_size, leaders, f_leaders, dim)
            f_pop = 1e+50 * np.ones((2 * pop_size,))
            # population = np.multiply(bound, np.random.uniform(-1, 1, (2*pop_size, dim)))
            # Why the population is not restarted? !!
            # To avoid having the origin as centroid

        indexes = np.argsort(f_pop)

        # Updating threshold
        min_step = np.maximum(alpha*d*(np.power((max_eval-count_eval)/max_eval, gamma)), 1e-05)
        max_step = 2*min_step

        # Population centroid
        centroid = np.tile(np.average(population[indexes[0:pop_size]], axis=0), (pop_size, 1))

        # Difference vectors
        dif = row_norm(np.subtract(centroid, leaders))

        # Difference vector scaling factor
        F = np.random.uniform(-max_step, max_step, (pop_size, ))

        # Orthogonal vectors # this may be wrong
        orthogonal = row_norm(np.random.normal(0, 1, (pop_size, dim)))
        orthogonal = row_norm(
            np.subtract(orthogonal, np.transpose(np.tile(np.sum(orthogonal.conj() * dif, axis=1), (dim, 1)))))

        # Orthogonal step scaling factor
        min_orth = np.sqrt(np.maximum(np.square(min_step)-np.square(F), 0))
        max_orth = np.sqrt(np.maximum(np.square(max_step)-np.square(F), 0))

        FO = np.transpose(np.random.uniform(min_orth, max_orth))

        population[indexes[pop_size:2 * pop_size], :] = \
            np.maximum(np.minimum(np.add(leaders,
                                         np.add(np.multiply(np.transpose(np.tile(F, (dim, 1))), dif),
                                                np.multiply(np.transpose(np.tile(FO, (dim, 1))), orthogonal))), bound),
                       -bound)
        f_pop[indexes[pop_size:2*pop_size]] = fun(population[indexes[pop_size:2*pop_size], :])
        count_eval = count_eval + pop_size

    leaders, f_leaders = merge_pops(population, f_pop, pop_size, leaders, f_leaders, dim)
    indexes = np.argsort(f_leaders)
    return leaders[indexes[0], :], f_leaders[indexes[0]]
