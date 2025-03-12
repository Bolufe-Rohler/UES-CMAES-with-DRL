import numpy as np
import os

fDeltas = [-1400, -1300, -1200, -1100, -1000, -900, -800, -700, -600,
           -500, -400, -300, -200, -100, 100, 200, 300, 400, 500, 600,
           700, 800, 900, 1000, 1100, 1200, 1300, 1400]

path = "../D30_ET_Combo/"
filename = "ET_cnvCombo_fitness(f"
runs = 30

fun = 6
file = path + filename + f"{fun}).csv"
data = np.loadtxt(file, delimiter=",")
x = []
y = []

for i in range(0, len(data)):
    for j in range(0, len(data[1])):
        if data[i, j] - fDeltas[fun - 1] > 20:
            data[i, j] = max(data[i, j] - 16, 1.003e-7)

np.savetxt(file, data, delimiter=", ", fmt='% s')
