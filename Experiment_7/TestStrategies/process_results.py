import os
import numpy as np

path = "./results/"

results = []  # np.zeros((24, 28))
best_errors = []
median_errors = []
for fun in range(1, 29):
    file = path + f"UES_CMAES_X_returns(f{fun}).csv"
    if os.path.exists(file):
        data = np.loadtxt(file, delimiter=", ")
        best_error = np.min(np.abs(data))
        average = np.abs(np.mean(data, 1))
        median_ave = np.median(average)
        sorted_id = np.argsort(average)
        rank = 1
        idx = 0
        part_res = list(sorted_id[:3])
        # while rank<4:
        #    part_res.append()
        results.append(part_res)
        best_errors.append(best_error)
        median_errors.append(median_ave)
        pass

# counting occurrences
occurrences = {}
for per_f in results:
    for f in per_f:
        if f in occurrences.keys():
            occurrences[f] = occurrences[f] + 1
        else:
            occurrences[f] = 1

occurrences = {k: v for k, v in sorted(occurrences.items(), key=lambda item: item[1])}
print(occurrences)

print(results)


# best errors
print("best errors")
print(best_errors)
print(median_errors)
