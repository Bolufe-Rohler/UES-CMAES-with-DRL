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


def ues_cmaes_X(fun, dim, max_eval, bound, state_size, start_point, decisions):  # NEEDS initial seed
    # decisions is a dictionary with the values for the design decisions of UES-CMAES
    eval_split = decisions['FE']  # 0.5 for 50:50; 0.9 for 90:10
    start_range = decisions['range']  # 0 for broad restart; 3 for focused restart
    sigma0 = decisions['sigma']  # 0.1 for very close search; 1 broad; 10 most search space
    gamma = decisions['gamma']
    alpha = decisions['alpha']
    iter_per_state = decisions['iters']
    cmaes_popsize = decisions['cma_pop']
    # MPS parameters
    # alpha = 0.1
    # gamma = 2
    d = np.sqrt(dim)*2*bound

    ues_eval = int(eval_split*max_eval)
    cmaes_eval = max_eval - ues_eval
    # iter_per_state = 30
    pop_size = int(ues_eval/(iter_per_state*state_size))

    states = np.zeros((state_size, dim))
    observations = -100 * np.ones((3*state_size+5, ))
    # observations are the updates on followers plus the restarts plus distance between best and worse for the 20 states
    # plus whether CMA-ES improved, relative improvement, distance advanced by CMAES from ues
    # distances from starting point for ues and for cmaes
    state_count = 0
    restarts_count = 0
    iter_fit_best = 1e+50
    iter_fit_worse = -1e+50
    iter_worse = np.zeros((1, dim))  # keeping the best and worse for calculating their distance
    iter_best = np.zeros((1, dim))
    updates_followers = 0

    # Initial population
    population = np.zeros((2*pop_size, dim))
    f_pop = 1e+50 * np.ones((2*pop_size, ))

    leaders = np.multiply(bound, np.random.uniform(-1, 1, (pop_size, dim)))
    if start_point is not None:
        leaders = leaders/(5*start_range + 4) + start_point
        d = np.sqrt(dim) * bound/(5*start_range+1)
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
        if current_iter % iter_per_state == 0:
            states[state_count] = leaders[0]

            # and now adding the restart counts
            observations[state_count] = restarts_count
            observations[state_size + state_count] = np.linalg.norm(iter_worse - iter_best)
            observations[2 * state_size + state_count] = updates_followers
            state_count += 1
            restarts_count = 0
            iter_fit_best = 1e+50
            iter_fit_worse = -1e+50
            updates_followers = 0

    leaders, f_leaders = merge_pops(population, f_pop, pop_size, leaders, f_leaders, dim)
    # indexes = np.argsort(f_leaders)

    x0 = leaders[0, :]
    f_ues = f_leaders[0]
    # optimum is expected to lie within about x0 +- 3*sigma0  ... so sigma0=15 covers most space assuming initial
    # solution is centered
    opts = cma.CMAOptions()
    es = cma.CMAEvolutionStrategy(x0, sigma0, {'bounds': [-bound, bound], 'maxfevals': cmaes_eval,
                                               'popsize': cmaes_popsize, 'verbose': -9})
    es.optimize(fun)

    observations[state_size - 1] = restarts_count
    observations[2*state_size - 1] = np.linalg.norm(iter_worse - iter_best)
    observations[3 * state_size - 1] = updates_followers

    # relative improvement by cmaes
    observations[3 * state_size] = 100*(es.result.fbest - f_ues)/max(abs(es.result.fbest), abs(f_ues))

    if es.result.fbest < f_ues:
        f_ues = es.result.fbest
        states[state_size-1, :] = es.result.xbest
        observations[3 * state_size + 1] = 1
    else:
        states[state_size - 1, :] = x0
        observations[3 * state_size + 1] = 0

    # distance moved by cmaes
    observations[3 * state_size + 2] = np.linalg.norm(x0 - es.result.xbest)


    # distance moved from initial solution for UES and cmaes
    if start_point is not None:
        observations[3 * state_size + 3] = np.linalg.norm(x0 - start_point)
        observations[3 * state_size + 4] = np.linalg.norm(es.result.xbest - start_point)
    else:
        observations[3 * state_size + 3] = -100  # if it was the first time
        observations[3 * state_size + 4] = -100  # if it was the first time

    return states, observations, f_ues