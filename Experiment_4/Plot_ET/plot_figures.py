import os
import numpy as np
import matplotlib.pyplot as plt


fDeltas = [-1400, -1300, -1200, -1100, -1000, -900, -800, -700, -600,
           -500, -400, -300, -200, -100, 100, 200, 300, 400, 500, 600,
           700, 800, 900, 1000, 1100, 1200, 1300, 1400]


path = "../D30_ET_Range/"
filename = "ET_cnvRange_fitness(f"
runs = 30

for fun in range(6, 21):
    file = path+filename+f"{fun}).csv"
    if os.path.exists(file):
        data = np.loadtxt(file, delimiter=", ")


        x = []
        y = []
        for i in range(len(data)):
            y.extend(np.abs(data[i, :]-fDeltas[fun-1]))
            x.extend(len(data[i, :]) * [i])

        fig, ax = plt.subplots()
        ax.scatter(x, y, s=2)
        ax.set_xlabel("Policies")
        ax.set_ylabel("Error from global optimum")
        ax.set_xticks(ticks=[0, 1, 2, 3, 4, 5, 6])
        ax.set_xticklabels(["local S.", "intermediate", "global", "uniform", "expl_expl", "3step e_e", "random"])

        # plt.show()
        plt.title(f"F{fun}")
        plt.savefig(path+filename+f"{fun}).png")
        plt.close()

