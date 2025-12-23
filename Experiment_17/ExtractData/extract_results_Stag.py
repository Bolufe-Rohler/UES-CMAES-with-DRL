import os
import numpy as np
import matplotlib.pyplot as plt


def rel_diff(x, y):
    if not max(y, x) == 0:
        return (x - y) / max(x, y)
    return 0


rel_diff = np.frompyfunc(rel_diff, 2, 1)

fDeltas = [-1400, -1300, -1200, -1100, -1000, -900, -800, -700, -600,
           -500, -400, -300, -200, -100, 100, 200, 300, 400, 500, 600,
           700, 800, 900, 1000, 1100, 1200, 1300, 1400]

path_algs = "../D100_benchmark_algs/"
path_std = "../../Experiment_17/D100_benchmark_ComboStag/"

filename_results = "comboStag_vs_algs_100d.csv"

filename_std = "comboStag_results(100D).csv"
results = np.zeros((28, 16))
data = np.loadtxt(path_std + filename_std, delimiter=",")
results[:, 0] = np.mean(data, axis=1)

filename_algs = "UES_CMAES_100d_1x.csv"
data = np.loadtxt(path_algs + filename_algs, delimiter=",")
results[:, 2] = np.mean(data, axis=1)

filename_algs = "UES_100d_1x.csv"
data = np.loadtxt(path_algs + filename_algs, delimiter=",")
results[:, 5] = np.mean(data, axis=1)

filename_algs = "CMAES_100d_1x.csv"
data = np.loadtxt(path_algs + filename_algs, delimiter=",")
results[:, 8] = np.mean(data, axis=1)

filename_algs = "PSO_100d_1x.csv"
data = np.loadtxt(path_algs + filename_algs, delimiter=",")
results[:, 11] = np.mean(data, axis=1)

filename_algs = "DE_100d_1x.csv"
data = np.loadtxt(path_algs + filename_algs, delimiter=",")
results[results < 1e-08] = 0
results[:, 14] = np.mean(data, axis=1)

results[results < 1e-08] = 0

# calculating relative differences
results[:, 3] = np.transpose(rel_diff(results[:, 0], results[:, 2]))
results[:, 6] = np.transpose(rel_diff(results[:, 0], results[:, 5]))
results[:, 9] = np.transpose(rel_diff(results[:, 0], results[:, 8]))
results[:, 12] = np.transpose(rel_diff(results[:, 0], results[:, 11]))
results[:, 15] = np.transpose(rel_diff(results[:, 0], results[:, 14]))


np.savetxt("../" + filename_results, results, delimiter=", ", fmt='% s')
