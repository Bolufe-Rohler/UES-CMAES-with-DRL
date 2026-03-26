import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read comparison file
df = pd.read_csv("exp21_dqn_vs_ddqn_per_function_comparison.csv")

# Expected columns:
# function
# mean_across_seeds_dqn
# mean_across_seeds_ddqn
# difference_dqn_minus_ddqn
# winner

functions = df["function"].to_numpy()
diffs = df["difference_dqn_minus_ddqn"].to_numpy()

# positive -> DDQN better
# negative -> DQN better
positive = diffs > 0
negative = diffs < 0
zero = np.isclose(diffs, 0.0)

plt.figure(figsize=(8, 4.8))
plt.axhline(0, linewidth=1)

plt.scatter(functions[positive], diffs[positive], label="DDQN better")
plt.scatter(functions[negative], diffs[negative], label="DQN better")
if np.any(zero):
    plt.scatter(functions[zero], diffs[zero], label="Tie")

for x, y in zip(functions, diffs):
    plt.plot([x, x], [0, y], linewidth=0.8)

plt.xticks(functions)
plt.xlabel("Benchmark function")
plt.ylabel("Difference in mean error (DQN - DDQN)")
plt.title("Per-function difference in optimization error (DQN - Double DQN)")
plt.legend()
plt.tight_layout()

plt.savefig("exp21_paired_difference.png", dpi=300, bbox_inches="tight")
plt.savefig("exp21_paired_difference.pdf", bbox_inches="tight")
plt.show()