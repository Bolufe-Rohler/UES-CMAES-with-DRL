import sys

import functions
import numpy as np
import scipy.optimize
import os

runs = 30
dim = 100
bound = 100

# Fitness of the global optima for the 28 functions of the CEC'13 benchmark
fDeltas = [-1400, -1300, -1200, -1100, -1000, -900, -800, -700, -600,
           -500, -400, -300, -200, -100, 100, 200, 300, 400, 500, 600,
           700, 800, 900, 1000, 1100, 1200, 1300, 1400]

# max_evals = 10_000 * dim  # like in the CEC'13 benchmark

cec_benchmark = functions.CEC_functions(dim)
bounds = dim * [(-bound, bound)]
popsize_DE = 50


# max_iter = int(max_evals / (popsize_DE * dim))


def obj_function(X):
    if len(X.shape) > 1:
        return cec_benchmark.Y_matrix(X, func_num)
    else:
        return cec_benchmark.Y(X, func_num)


# #used for different functions
# fun_num = 0
# results = np.zeros(29)
# for func_num in range(1, 29):
#     fun_num = func_num
#     for run in range(0, runs):
#         de = scipy.optimize.differential_evolution(obj_function, bounds=bounds, maxiter=max_iter, polish=False)
#         results[func_num-1] += de.fun
#
#     results[func_num-1] = results[func_num-1] / runs
#     print(f"Function {func_num}, DE result (error respect to the global optimum): {(results[func_num-1]-fDeltas[func_num-1]):.2E}")
#
# np.save(f"results_de.np", results)

max_mult = 1
max_evals = max_mult*dim * 10_000
results_file = f"DE_100d_{max_mult}x.csv"
if os.path.exists(results_file):
    # load existing data
    file = open(results_file, "r")
    results = np.loadtxt(file, delimiter=',')
    file.close()
else:
    results = np.zeros((28, 30))

max_iter = int(max_evals / (popsize_DE * dim))
for func_num in range(1, 29):
    print(f"For function {func_num} ...")
    for run in range(0, runs):
        de = scipy.optimize.differential_evolution(obj_function, bounds=bounds, maxiter=max_iter, polish=False)
        results[func_num - 1, run] = (de.fun - fDeltas[func_num - 1])

    np.savetxt(results_file, results, delimiter=", ", fmt='% s')
