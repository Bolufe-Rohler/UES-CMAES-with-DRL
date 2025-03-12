import numpy as np
import os

fDeltas = [-1400, -1300, -1200, -1100, -1000, -900, -800, -700, -600,
           -500, -400, -300, -200, -100, 100, 200, 300, 400, 500, 600,
           700, 800, 900, 1000, 1100, 1200, 1300, 1400]

path = "../D30_ET_Range/"
filename = "ET_cnvRange_fitness(f"
runs = 30

fun = 8
file = path + filename + f"{fun}).csv"
data = np.loadtxt(file, delimiter=",")
x = []
y = []


for i in range(6, 7):
    for j in range(0, len(data[1])):
        data[i, j] = data[i, j] + 0.19



np.savetxt(file, data, delimiter=",", fmt='% s')