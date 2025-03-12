import numpy as np
# from functions import *
import cma


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


def ues_cmaes_rl(fun, dim, max_eval, bound, observ_size, start_point, action):  # NEEDS initial seed

    # MPS parameters
    alpha = 0.1
    gamma = 1
    d = np.sqrt(dim)*2*bound

    ues_eval = int(0.9*max_eval)
    cmaes_eval = max_eval - ues_eval
    iter_per_observ = 30
    pop_size = int(ues_eval / (iter_per_observ * observ_size))

    # state = np.zeros(1, dim)  # The state is the final solution
    observations = 1e+50 * np.ones((3 * observ_size,))  # observations are the fitness of best solutions plus the restarts
    observ_count = 0
    restarts_count = 0
    iter_fit_best = 1e+50
    iter_fit_worse = -1e+50
    iter_worse = np.zeros((1, dim)) # keeping the best and worse for calculating their distance
    iter_best = np.zeros((1, dim))
    updates_followers = 0

    # Initial population
    population = np.zeros((2*pop_size, dim))
    f_pop = 1e+50 * np.ones((2*pop_size, ))

    leaders = np.multiply(bound, np.random.uniform(-1, 1, (pop_size, dim)))
    if start_point is not None:
        leaders = leaders/(5*action + 4) + start_point
        d = np.sqrt(dim) * bound/(5*action+1)
    f_leaders = fun(leaders)
    count_eval = pop_size

    current_median = np.median(f_leaders)

    current_iter = 0
    while count_eval < ues_eval:
        new_median = np.median(f_pop)
        if current_median > new_median:
            current_median = new_median

            leaders, f_leaders = merge_pops(population, f_pop, pop_size, leaders, f_leaders, dim)
            f_pop = 1e+50 * np.ones((2 * pop_size,))
            restarts_count += 1
            # population = np.multiply(bound, np.random.uniform(-1, 1, (2*pop_size, dim)))
            # Why the population is not restarted? !!
            # To avoid having the origin as centroid

        indexes = np.argsort(f_pop)


        # Updating threshold
        min_step = np.maximum(alpha*d*(np.power((ues_eval-count_eval)/ues_eval, gamma)), 1e-05)
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


        ### best and worse update for data
        if f_pop[indexes[0]] < iter_fit_best:
            iter_fit_best = f_pop[indexes[0]]
            iter_best = population[indexes[0]]

        if f_pop[indexes[2*pop_size-1]] != 1e+50 and f_pop[indexes[2*pop_size-1]] > iter_fit_worse:
            iter_fit_worse = f_pop[indexes[2*pop_size-1]]
            iter_worse = population[indexes[2*pop_size-1]]

        ### Updates of new Followers kept for data
        updates_followers += np.sum(np.median(f_pop[indexes[1:pop_size]]) > f_pop[indexes[pop_size+1: 2*pop_size]])

        current_iter += 1
        if current_iter % iter_per_observ == 0:

            # and now adding the restart counts
            observations[observ_count] = restarts_count
            observations[observ_size + observ_count] = np.linalg.norm(iter_worse - iter_best)
            observations[2 * observ_size + observ_count] = updates_followers
            observ_count += 1
            restarts_count = 0
            iter_fit_best = 1e+50
            iter_fit_worse = -1e+50
            updates_followers = 0

    leaders, f_leaders = merge_pops(population, f_pop, pop_size, leaders, f_leaders, dim)

    x0 = leaders[0, :]
    f_ues = f_leaders[0]
    es = cma.CMAEvolutionStrategy(x0, 0.1, {'bounds': [-bound, bound], 'maxfevals': cmaes_eval,
                                          'verbose': -9})
    es.optimize(fun)

    observations[observ_size - 1] = restarts_count
    observations[2 * observ_size - 1] = np.linalg.norm(iter_worse - iter_best)
    observations[3 * observ_size - 1] = updates_followers
    if es.result.fbest < f_ues:
        f_ues = es.result.fbest
        state = es.result.xbest
    else:
        state = x0

    return state, observations, f_ues