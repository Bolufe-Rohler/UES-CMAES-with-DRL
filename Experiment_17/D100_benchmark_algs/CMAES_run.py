import numpy as np
import cma
import functions
import sys
import os

fDeltas = [-1400, -1300, -1200, -1100, -1000, -900, -800, -700, -600,
           -500, -400, -300, -200, -100, 100, 200, 300, 400, 500, 600,
           700, 800, 900, 1000, 1100, 1200, 1300, 1400]

runs = 30
dim = 100
cec_benchmark = functions.CEC_functions(dim)


def obj_function(X):
    if len(X.shape) > 1:
        return cec_benchmark.Y_matrix(X, func_num)
    else:
        return cec_benchmark.Y(X, func_num)


bound = 100
bounds = dim * [(-bound, bound)]

max_mult = 1
max_evals = max_mult*10_000 * dim  # like in the CEC'13 benchmark
results_file = f"CMAES_100d_{max_mult}x.csv"
if os.path.exists(results_file):
    # load existing data
    file = open(results_file, "r")
    results = np.loadtxt(file, delimiter=',')
    file.close()
else:
    results = np.zeros((28, 30))


for func_num in range(1, 29):
    print(f"For function {func_num} ...")
    for run in range(0, runs):
        x0 = bound - 2 * bound * np.random.rand(dim)
        es = cma.CMAEvolutionStrategy(x0, 50, {'bounds': [-bound, bound], 'maxfevals': max_evals, 'verbose': -9})
        es.optimize(obj_function)
        results[func_num-1, run] = (es.result.fbest - fDeltas[func_num - 1])

    np.savetxt(results_file, results, delimiter=", ", fmt='% s')