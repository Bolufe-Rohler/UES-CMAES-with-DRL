import functions
import numpy as np
from Standard_PSO import *
import pickle
import sys
import os

runs = 30
dim = 30
bound = 100

swarmsize = 50

# Fitness of the global optima for the 28 functions of the CEC'13 benchmark
fDeltas = [-1400, -1300, -1200, -1100, -1000, -900, -800, -700, -600,
           -500, -400, -300, -200, -100, 100, 200, 300, 400, 500, 600,
           700, 800, 900, 1000, 1100, 1200, 1300, 1400]


# results = np.zeros(29)
# for func_num in range(1, 29):
#     for run in range(0, runs):
#         pso = ParticleSwarmOptimizer(func_num)
#         best_fit, best = pso.optimize()
#         results[func_num-1] += best_fit
#
#     results[func_num-1] = results[func_num-1] / runs
#     print(f"Function {func_num}, result (error respect to the global optimum): {(results[func_num-1]-fDeltas[func_num-1]):.2E}")
#
# np.save(f"results_pso.np", results)


# func_num = int(sys.argv[1])
max_mult = 1
max_evals = max_mult*10_000 * dim  # like in the CEC'13 benchmark
results_file = f"PSO_{max_mult}x.csv"
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
        pso = ParticleSwarmOptimizer(func_num, max_evals, 50)
        best_fit, best = pso.optimize()
        results[func_num-1, run] = (best_fit - fDeltas[func_num - 1])

    np.savetxt(results_file, results, delimiter=", ", fmt='% s')
