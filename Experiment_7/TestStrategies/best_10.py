if action == 0:
    decisions = {'FE': 0.9, 'range': 1, 'gamma': 1, 'sigma': 0.1, 'alpha': 0.1, 'cma_pop': 15, 'iters': 30}
elif action == 3:
    decisions = {'FE': 0.9, 'range': 0, 'gamma': 2, 'sigma': 10, 'alpha': 0.1, 'cma_pop': 30, 'iters': 30}
elif action == 6:
    decisions = {'FE': 0.9, 'range': 2, 'gamma': 2, 'sigma': 0.1, 'alpha': 0.05, 'cma_pop': 15, 'iters': 40}
elif action == 10:
    decisions = {'FE': 0.5, 'range': 3, 'gamma': 1, 'sigma': 10, 'alpha': 0.1, 'cma_pop': 45, 'iters': 50}
elif action == 11:
    decisions = {'FE': 0.5, 'range': 0, 'gamma': 2, 'sigma': 10, 'alpha': 0.1, 'cma_pop': 15, 'iters': 30}
elif action == 12:
    decisions = {'FE': 0.9, 'range': 0, 'gamma': 3, 'sigma': 10, 'alpha': 0.1, 'cma_pop': 15, 'iters': 30}
elif action == 13:
    decisions = {'FE': 0.9, 'range': 0, 'gamma': 1, 'sigma': 1, 'alpha': 0.1, 'cma_pop': 30, 'iters': 40}
elif action == 15:
    decisions = {'FE': 0.9, 'range': 0, 'gamma': 3, 'sigma': 0.1, 'alpha': 0.1, 'cma_pop': 15, 'iters': 40}
elif action == 19:
    decisions = {'FE': 0.5, 'range': 4, 'gamma': 1, 'sigma': 10, 'alpha': 0.1, 'cma_pop': 30, 'iters': 30}
elif action == 20:
    decisions = {'FE': 0.5, 'range': 4, 'gamma': 2, 'sigma': 1, 'alpha': 0.1, 'cma_pop': 15, 'iters': 40}
elif action == 22:
    decisions = {'FE': 0.5, 'range': 1, 'gamma': 1, 'sigma': 1, 'alpha': 0.1, 'cma_pop': 45, 'iters': 50}
else:
    decisions = {'FE': 0.5, 'range': 2, 'gamma': 2, 'sigma': 10, 'alpha': 0.05, 'cma_pop': 15, 'iters': 30}
