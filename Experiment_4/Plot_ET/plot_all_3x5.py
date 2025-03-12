import os
import numpy as np
import matplotlib.pyplot as plt

fDeltas = [-1400, -1300, -1200, -1100, -1000, -900, -800, -700, -600,
           -500, -400, -300, -200, -100, 100, 200, 300, 400, 500, 600,
           700, 800, 900, 1000, 1100, 1200, 1300, 1400]

path = "../D30_ET_Range/"
filename = "ET_cnvRange_fitness(f"

# Adjusted layout: 3 rows x 5 columns
fig_all, ax = plt.subplots(3, 5, figsize=(18, 12), constrained_layout=True)
bot_lim = [0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7]
top_lim = [80, 105, 22, 25, 7, 70, 70, 150, 4000, 4000, 0.3, 65, 80, 3, 14]

for fun in range(6, 21):
    file = path + filename + f"{fun}).csv"
    if os.path.exists(file):
        data = np.loadtxt(file, delimiter=",")

        x = []
        y = []
        for i in range(len(data)):
            y.extend(np.abs(data[i, :] - fDeltas[fun - 1]))
            x.extend(len(data[i, :]) * [i])

        row = (fun - 6) // 5
        col = (fun - 6) % 5
        ax[row, col].scatter(x, y, s=2)

        ax[row, col].set_xticks(ticks=[0, 1, 2, 3, 4, 5, 6])
        ax[row, col].set_xticklabels(["p1", "p2", "p3", "p4", "p5", "p6", "p7"])

        ax[row, col].set_title(f"F{fun}", y=1.0, pad=5)
        ax[row, col].set_ylim(bot_lim[fun - 6], top_lim[fun - 6])

# Add global labels with better positioning
fig_all.suptitle('Error from the Global Optimum by Policy', fontsize=16, y=0.98)
fig_all.text(0.5, 0.04, 'Policies', ha='center', fontsize=12)
fig_all.text(0.04, 0.5, 'Error from the Global Optimum', va='center', rotation='vertical', fontsize=12)

# Ensure proper layout and save the figure
plt.savefig(path + filename + f"_all_corrected.png")
plt.show()
plt.close()
