import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import ScalarFormatter

fDeltas = [-1400, -1300, -1200, -1100, -1000, -900, -800, -700, -600,
           -500, -400, -300, -200, -100, 100, 200, 300, 400, 500, 600,
           700, 800, 900, 1000, 1100, 1200, 1300, 1400]

path = "../DQN_rdnplus_Combo/"
filename = "DQN_rdnplus_Combo"
filename_loss = "DQN_rdn_cnvCombo_loss(f"
filename_re = "DQN_rdn_cnvCombo_returns(f"


fig_all, ax = plt.subplots(6, 2, figsize=(8, 15))
fig = 3
#bot_lim = [-10, -0.025, -18, -2600, -0.25, -3]
#top_lim = [0, 0, 0, -500, 0, -0.5]
bot_lim = [-1e1, -2.5e-2, -1.8e1, -2.6e3, -2.5e-1, -3e0]
top_lim = [0e0, 0e0, 0e0, -5e2, 0e0, -5e-1]
# top_lim = [80,105,22, 25,7,70, 70,150,4000, 4000,0.3,65, 80,3,14]

if fig == 1:
    functions = [6, 10, 11]
elif fig==2:
    functions = [14, 16, 19]  #   #
else:
    functions = [6, 10, 11, 14, 16, 19]
for idx in range(6):
    fun = functions[idx]
    file_loss = path + filename_loss + f"{fun}).csv"
    file_re = path + filename_re + f"{fun}).csv"
    if os.path.exists(file_loss):
        loss = np.loadtxt(file_loss, delimiter=",")
        loss = loss[0:100]
        rewards = np.loadtxt(file_re, delimiter=",")
        rewards = rewards[0:50]

        col = (fun - 6) % 3
        row = (fun - 6) // 3
        ax[idx, 0].plot(rewards)
        ax[idx, 1].plot(loss)
        # ax[row, col].set_xlabel("Policies")
        # ax[row, col].set_ylabel("Error from global optimum")
        # ax[idx, 0].set_xlim(0, 25000)
        # ax[idx, 1].set_xlim(0, 25000)
        ax[idx, 0].set_xticks(ticks=[0, 10, 20, 30, 40, 50])
        ax[idx, 0].set_xticklabels(["0", "5000", "10000", "15000", "20000", "25000"])
        ax[idx, 1].set_yscale("log")
        ax[idx, 1].set_xticks(ticks=[0, 20, 40, 60, 80, 100])
        ax[idx, 1].set_xticklabels(["0", "5000", "10000", "15000", "20000", "25000"])


        # plt.show()
        if idx ==0 :
            ax[idx, 0].set_title(f"Reward                                 "
                                 f"F{fun}                                Loss", x=1, y=1, pad=8)

        else:
            ax[idx, 0].set_title(f"                                              "
                                 f"F{fun}                                ", x=1, y=1, pad=8)


        ax[idx, 0].set_ylim(bot_lim[idx], top_lim[idx])

        # Set y-axis to scientific notation
        ax[idx, 0].yaxis.set_major_formatter(ScalarFormatter())
        ax[idx, 0].ticklabel_format(axis='y', style='sci')
        
        # fig_all.set_xlabel("Policies")

# fig_all.text(0.5, 0.08, 'Policies', ha='center')
# fig_all.text(0.08, 0.5, 'Error from the global optimum', va='center', rotation='vertical')
plt.savefig(path + filename + f"_fig{fig}.png")
plt.close()
