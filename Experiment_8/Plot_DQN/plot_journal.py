import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# Define the function names for labels
function_labels = {
    6: "F6: Rotated Rosenbrock’s",
    10: "F10: Rotated Weierstrass",
    11: "F11: Rastrigin",
    14: "F14: Schwefel's",
    16: "F16: Rotated Katsuura",
    19: "F19: Expanded Griewank’s\nplus Rosenbrock’s"
}


path = "../DQN_rdnplus_Ran/"
filename = "DQN_rdnplus_Ran"
filename_loss = "DQN_rdn_cnvRan_loss(f"
filename_re = "DQN_rdn_cnvRan_returns(f"

# Create figure with 3 rows and 4 columns (2 plots per function)
fig_all, ax = plt.subplots(3, 4, figsize=(16, 12),  sharey=False)
fig = 3
bot_lim = [-10,-0.025,-18,  -2600,-0.25,-3]
top_lim = [0,0,0,  -500,0,-0.5]

# Select the functions to plot
functions = [6, 10, 11, 14, 16, 19]

# Add column titles for the first row (both first and second function)
for col in range(2):
    ax[0, col * 2].set_title("Reward", fontsize=14, fontweight='bold')
    ax[0, col * 2 + 1].set_title("Loss", fontsize=14, fontweight='bold')

for idx, fun in enumerate(functions):
    file_loss = os.path.join(path, f"{filename_loss}{fun}).csv")
    file_re = os.path.join(path, f"{filename_re}{fun}).csv")

    if os.path.exists(file_loss) and os.path.exists(file_re):
        # Load data and limit samples to first 50 for rewards and first 100 for loss
        rewards = np.loadtxt(file_re, delimiter=",")[:50]  # Limit to 50 samples
        loss = np.loadtxt(file_loss, delimiter=",")[:100]  # Limit to 100 samples

        # Calculate subplot positions
        row, col = divmod(idx, 2)

        # Plot rewards (first 50 samples)
        ax[row, col * 2].plot(rewards, color='b')

        # Ensure the x-axis covers the full range of the reward array
        #ax[row, col * 2].set_xlim(0, 50)  # Force axis to match 50 samples

        # Set y-axis limits and format
        ax[row, col * 2].set_ylim(bot_lim[idx], top_lim[idx])

        # Add label inside reward plot (bottom-right corner)
        ax[row, col * 2].text(0.95, 0.05, function_labels[fun],
                              transform=ax[row, col * 2].transAxes,
                              fontsize=10, ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.7))

        # Plot loss (first 100 samples)
        ax[row, col * 2 + 1].plot(loss, color='r')
        ax[row, col * 2 + 1].set_yscale("log")

        # Set x-axis ticks and labels based on fixed sample limits
        ax[row, col * 2].set_xticks(ticks=[0, 10, 20, 30, 40, 50])
        ax[row, col * 2].set_xticklabels(["0", "5000", "10000", "15000", "20000", "25000"])

        ax[row, col * 2 + 1].set_xticks(ticks=[0, 20, 40, 60, 80, 100])
        ax[row, col * 2 + 1].set_xticklabels(["0", "5000", "10000", "15000", "20000", "25000"])

    else:
        print(f"Files for function F{fun} not found: {file_loss} or {file_re}")
# Add title to the entire figure
plt.suptitle("FinalRange Environment", fontsize=16, fontweight='bold')

# Adjust layout to minimize border space and save figure with correct filename
plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05, hspace=0.3, wspace=0.2)
plt.savefig(path + filename + "_ITOR.png")
plt.close()
