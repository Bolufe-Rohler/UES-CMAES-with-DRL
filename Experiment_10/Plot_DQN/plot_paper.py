import os
import numpy as np
import matplotlib.pyplot as plt

fDeltas = [-1400, -1300, -1200, -1100, -1000, -900, -800, -700, -600,
           -500, -400, -300, -200, -100, 100, 200, 300, 400, 500, 600,
           700, 800, 900, 1000, 1100, 1200, 1300, 1400]

path = "../D30_all_Combo_std/"
filename = "D30_all_Combo_std"
filename_loss = path + "DQN_ComboStd_loss.csv"
filename_re = path +"DQN_ComboStd_returns.csv"


fig_all, ax = plt.subplots(1, 2, figsize=(10, 2.5))

#fig = 1
bot_lim = [-10,-0.025,-18,  -2600,-0.25,-3]
top_lim = [0,0,0,  -500,0,-0.5]
# top_lim = [80,105,22, 25,7,70, 70,150,4000, 4000,0.3,65, 80,3,14]

loss = np.loadtxt(filename_loss, delimiter=",")
loss = loss[0:200]
rewards = np.loadtxt(filename_re, delimiter=",")
rewards = rewards[0:100]

ax[0].set_xticks(ticks=[0, 20, 40, 60, 80, 100])
ax[0].set_xticklabels(["0", "20000", "40000", "60000", "80000", "100000"])
ax[1].set_yscale("log")
ax[1].set_xticks(ticks=[0, 40, 80, 120, 160, 200])
ax[1].set_xticklabels(["0", "20000", "40000", "60000", "80000", "100000"])

ax[0].set_title(f"Reward                          "
                     f"Entire benchmark (standard reward)                     Loss", x=1, y=1, pad=8)

ax[0].plot(rewards, color='b')
ax[1].plot(loss, color='r')

plt.savefig(path + filename + f"std_paper.png")
plt.close()
