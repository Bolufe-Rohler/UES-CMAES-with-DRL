import numpy as np
import cma


_EPS = 1e-12


def row_norm(x: np.ndarray, eps: float = _EPS) -> np.ndarray:
    """
    Row-wise normalization with epsilon protection to avoid division by zero.
    """
    norms = np.sqrt(np.square(x).sum(axis=1, keepdims=True))
    norms = np.maximum(norms, eps)
    return x / norms


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


def ues_cmaes_X(
    fun,
    dim,
    max_eval,
    bound,
    state_size,
    start_point,
    decisions,
    include_single_run_features: bool = True,
):
    """
    UES-CMA-ES hybrid rollout used by the RL environment.

    Returns:
      states: (state_size, dim)
      observations: (3*state_size + (5 if include_single_run_features else 0),)
      f_ues: best fitness achieved after CMA-ES refinement (or UES if better)
    """
    # decisions is a dictionary with the values for the design decisions of UES-CMAES
    eval_split = decisions["FE"]       # 0.5 for 50:50; 0.9 for 90:10
    start_range = decisions["range"]   # 0 broad restart; 3 focused restart
    sigma0 = decisions["sigma"]        # CMA-ES initial step-size
    gamma = decisions["gamma"]
    alpha = decisions["alpha"]
    iter_per_state = decisions["iters"]
    cmaes_popsize = decisions["cma_pop"]

    d = np.sqrt(dim) * 2 * bound

    ues_eval = int(eval_split * max_eval)
    cmaes_eval = max_eval - ues_eval
    pop_size = int(ues_eval / (iter_per_state * state_size))
    pop_size = max(pop_size, 2)  # safety

    states = np.zeros((state_size, dim))

    base_obs_dim = 3 * state_size
    single_obs_dim = 5 if include_single_run_features else 0
    obs_dim = base_obs_dim + single_obs_dim
    observations = -100.0 * np.ones((obs_dim,), dtype=np.float64)

    state_count = 0
    restarts_count = 0
    iter_fit_best = 1e50
    iter_fit_worse = -1e50
    iter_worse = np.zeros((dim,))
    iter_best = np.zeros((dim,))
    updates_followers = 0

    # Initial population
    population = np.zeros((2 * pop_size, dim))
    f_pop = 1e50 * np.ones((2 * pop_size,))

    leaders = np.multiply(bound, np.random.uniform(-1, 1, (pop_size, dim)))
    if start_point is not None:
        leaders = leaders / (5 * start_range + 4) + start_point
        d = np.sqrt(dim) * bound / (5 * start_range + 1)

    f_leaders = fun(leaders)
    count_eval = pop_size

    current_median = np.median(f_leaders)
    current_iter = 0

    while count_eval < ues_eval:
        new_median = np.median(f_pop)
        if current_median > new_median:
            current_median = new_median
            leaders, f_leaders = merge_pops(population, f_pop, pop_size, leaders, f_leaders, dim)
            f_pop = 1e50 * np.ones((2 * pop_size,))
            restarts_count += 1

        indexes = np.argsort(f_pop)

        # Updating threshold
        frac = (ues_eval - count_eval) / max(ues_eval, 1)
        min_step = np.maximum(alpha * d * (np.power(frac, gamma)), 1e-05)
        max_step = 2 * min_step

        # Population centroid
        centroid = np.tile(np.average(population[indexes[0:pop_size]], axis=0), (pop_size, 1))

        # Difference vectors (safe)
        dif = row_norm(np.subtract(centroid, leaders))

        # Difference vector scaling factor
        F = np.random.uniform(-max_step, max_step, (pop_size,))

        # Orthogonal vectors (safe)
        orthogonal = row_norm(np.random.normal(0, 1, (pop_size, dim)))
        proj = np.sum(orthogonal * dif, axis=1, keepdims=True)  # (pop_size, 1)
        orthogonal = row_norm(orthogonal - proj * dif)

        # Orthogonal step scaling factor
        min_orth = np.sqrt(np.maximum(np.square(min_step) - np.square(F), 0))
        max_orth = np.sqrt(np.maximum(np.square(max_step) - np.square(F), 0))
        FO = np.random.uniform(min_orth, max_orth, size=(pop_size,))

        step_vec = (F[:, None] * dif) + (FO[:, None] * orthogonal)
        new_points = leaders + step_vec
        new_points = np.clip(new_points, -bound, bound)

        population[indexes[pop_size:2 * pop_size], :] = new_points
        f_pop[indexes[pop_size:2 * pop_size]] = fun(new_points)
        count_eval += pop_size

        # best and worst update for observation diagnostics
        if f_pop[indexes[0]] < iter_fit_best:
            iter_fit_best = f_pop[indexes[0]]
            iter_best = population[indexes[0]].copy()

        worst_idx = indexes[2 * pop_size - 1]
        if f_pop[worst_idx] != 1e50 and f_pop[worst_idx] > iter_fit_worse:
            iter_fit_worse = f_pop[worst_idx]
            iter_worse = population[worst_idx].copy()

        # follower updates kept for data
        # (note: this is your original logic; kept intact)
        updates_followers += np.sum(np.median(f_pop[indexes[1:pop_size]]) > f_pop[indexes[pop_size + 1: 2 * pop_size]])

        current_iter += 1
        if current_iter % iter_per_state == 0 and state_count < state_size:
            states[state_count] = leaders[0]

            observations[state_count] = restarts_count
            observations[state_size + state_count] = np.linalg.norm(iter_worse - iter_best)
            observations[2 * state_size + state_count] = updates_followers

            state_count += 1
            restarts_count = 0
            iter_fit_best = 1e50
            iter_fit_worse = -1e50
            updates_followers = 0

    leaders, f_leaders = merge_pops(population, f_pop, pop_size, leaders, f_leaders, dim)

    x0 = leaders[0, :]
    f_ues = f_leaders[0]

    es = cma.CMAEvolutionStrategy(
        x0,
        sigma0,
        {
            "bounds": [-bound, bound],
            "maxfevals": cmaes_eval,
            "popsize": cmaes_popsize,
            "verbose": -9,
        },
    )
    es.optimize(fun)

    # Fill the last checkpoint slot (your original behavior)
    last_idx = state_size - 1
    observations[last_idx] = restarts_count
    observations[state_size + last_idx] = np.linalg.norm(iter_worse - iter_best)
    observations[2 * state_size + last_idx] = updates_followers

    if include_single_run_features:
        base = 3 * state_size

        denom = max(abs(es.result.fbest), abs(f_ues), _EPS)
        rel_impr = 100.0 * (es.result.fbest - f_ues) / denom

        # relative improvement by cmaes (safe)
        observations[base + 0] = rel_impr

        if es.result.fbest < f_ues:
            f_ues = es.result.fbest
            states[last_idx, :] = es.result.xbest
            observations[base + 1] = 1.0
        else:
            states[last_idx, :] = x0
            observations[base + 1] = 0.0

        # distance moved by cmaes
        observations[base + 2] = np.linalg.norm(x0 - es.result.xbest)

        # distance moved from initial solution for UES and cmaes
        if start_point is not None:
            observations[base + 3] = np.linalg.norm(x0 - start_point)
            observations[base + 4] = np.linalg.norm(es.result.xbest - start_point)
        else:
            observations[base + 3] = -100.0
            observations[base + 4] = -100.0
    else:
        # still update best if CMA improved, but no extra obs slots
        if es.result.fbest < f_ues:
            f_ues = es.result.fbest
            states[last_idx, :] = es.result.xbest
        else:
            states[last_idx, :] = x0

    # FINAL SAFETY: never return NaN/Inf observations
    observations = np.nan_to_num(
        observations,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float64)

    return states, observations, f_ues
