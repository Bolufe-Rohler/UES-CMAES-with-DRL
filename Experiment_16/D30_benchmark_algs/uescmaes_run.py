import sys
import functions
from ues_cmaes import *
import numpy as np
import os
import csv

# These are the fitness values of the global optima for the 28 functions in the benchmark.
fDeltas = [-1400, -1300, -1200, -1100, -1000, -900, -800, -700, -600,
           -500, -400, -300, -200, -100, 100, 200, 300, 400, 500, 600,
           700, 800, 900, 1000, 1100, 1200, 1300, 1400]

runs = 30
dim = 30
cec_benchmark = functions.CEC_functions(dim)

bound = 100
bounds = dim * [(-bound, bound)]


def obj_function(X):
    if len(X.shape) > 1:
        return cec_benchmark.Y_matrix(X, func_num)
    else:
        return cec_benchmark.Y(X, func_num)


max_mult = 1
max_evals = max_mult*10_000 * dim  # like in the CEC'13 benchmark
results_file = f"UES_CMAES_{max_mult}x.csv"
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
        best_solution, best_fitness = ues_cmaes(obj_function, dim=dim, max_eval=max_evals,
                                                pop_size=100, bound=bound)
        results[func_num-1, run] = (best_fitness - fDeltas[func_num - 1])

    np.savetxt(results_file, results, delimiter=", ", fmt='% s')
