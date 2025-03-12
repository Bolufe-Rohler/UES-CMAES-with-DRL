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

path_norm = "../D30_benchmark_Combo_norm/"
path_std = "../../Experiment_12/D30_benchmark_Combo_std/"

'''
filename = "combo_norm_results(35000).csv"
filename_results = "combo_norm_results(35000).csv"
results = np.zeros((28, 3))
data = np.loadtxt(path + filename, delimiter=",")
results[:, 0] = np.mean(data, axis=1)
results[:, 1] = np.min(data, axis=1)
results[:, 2] = np.max(data, axis=1)
results[results < 1e-08] = 0
print(results)
'''

filename_std = "combo_std_results.csv"
filename_results = "combo_results_std_vs_norm_new.csv"
results_std = np.zeros((28, 4))
data_std = np.loadtxt(path_std + filename_std, delimiter=",")
results_std[:, 0] = np.mean(data_std, axis=1)
results_std[:, 1] = np.min(data_std, axis=1)
results_std[:, 2] = np.max(data_std, axis=1)
results_std[:, 3] = np.std(data_std, axis=1)
results_std[results_std < 1e-08] = 0
print(results_std)

filename_norm = "combo_norm_results(103000).csv"
filename_results_norm = "combo_norm_results(103000).csv"

results_norm = np.zeros((28, 4))
data_norm = np.loadtxt(path_norm + filename_norm, delimiter=",")
results_norm[:, 0] = np.mean(data_norm, axis=1)
results_norm[:, 1] = np.min(data_norm, axis=1)
results_norm[:, 2] = np.max(data_norm, axis=1)
results_norm[:, 3] = np.std(data_norm, axis=1)
results_norm[results_norm < 1e-08] = 0
print(results_norm)

rel_dif = rel_diff(results_std[:, 0], results_norm[:, 0])
print(rel_dif*100)
print(np.mean(rel_dif)*100)

final_result = np.zeros((28, 11))
final_result[:, 0:4] = results_std[:, 0:4]
final_result[:, 5:9] = results_norm[:, 0:4]
final_result[:, 9] = np.transpose(rel_dif)


np.savetxt("../" + filename_results, final_result, delimiter=", ", fmt='% s')
