# Deep Reinforcement Learning for Multi-Restart Strategies in UES-CMA-ES

This repository contains various experiments that explore deep reinforcement learning (DRL) for multi-restart strategies in the UES-CMA-ES metaheuristic hybrid. Each experiment folder includes code, scripts, and partial results or logs. Below is an overview of what each experiment investigates and the primary outcomes.

For experiments that rely on trained reinforcement learning agents, the trained models are available in the `policy/` directory inside the corresponding experiment folder. These policies can be loaded directly for evaluation and reproducibility without retraining.

## Experiment 1

**Goal:** Test the initial Fit100-Act100 approach on a small set of functions (e.g., F6, F9, F11, etc.) where the environment tries to restart from one of 100 recorded solutions during the run.

### Description
- The agent picks from 100 snapshots of the best fitness observed at equidistant intervals.
- No parameter changes—only restart positions are different.

### Outcome
- Results indicate little or no improvement in reward.
- The policy does not meaningfully differentiate among the 100 possible restart points.
- Concluded that focusing solely on the restart solution (without parameter changes) is insufficient.

## Experiment 2

**Goal:** Investigate whether reducing the action space (from 100 possible restart indices down to 50 or 10) helps.

### Description
- Same environment structure as Experiment 1, but with Fit100-Act50 and Fit100-Act10.
- The agent can only pick among 50 or 10 restart snapshots instead of 100.

### Outcome
- Performance remains poor, with random-like reward curves.
- Merely shrinking the action space does not fix the limitations of the environment.

## Experiment 3

**Goal:** Extend the analysis of the initial approach by trying different policies (manually defined) rather than training an agent.

### Description
- Same Fit100-Act100 environment, but this time testing preset policies (e.g., always pick the first snapshot, always pick the last, pick midpoints, etc.).
- No DRL training—just direct policy evaluation.

### Outcome
- Independent of which policy is used, the final results are nearly identical.
- Suggests the environment itself (i.e., restarting mid-convergence without parameter changes) is fundamentally unhelpful in improving outcomes.

## Experiment 4

**Goal:** Explore a new environment (FinalRange) that restarts from the final solution but varies the range for sampling new UES solutions.

### Description
- The range for sampling around the best solution is controlled by an action (0 to 4).
- Different policies define how often to pick a wide vs. narrow range.

### Outcome
- Policies produce clearly different final errors—some do better with consistently wide restarts, others with narrow.
- Demonstrates that controlling the restart range can differentiate policies and improve results over the previous approach.

## Experiment 5

**Goal:** Introduce another environment (FinalRate) that focuses on adjusting the convergence rate gamma (γ) of UES at each restart.

### Description
- Restarts always happen from the final solution, but the agent/policy sets γ in {0,1,2,3,4} controlling how fast UES converges.
- Policies again tested with a similar approach to Experiment 4.

### Outcome
- Some variations among policies are observed, though not as pronounced as changing the restart range.
- Convergence rate tuning alone yields only moderate improvements compared to FinalRange.

## Experiment 6

**Goal:** Combine multiple parameters (range, γ, FEs, etc.) into one environment (FinalCombo) and see how different parameter combos affect policy outcomes.

### Description
- Each action corresponds to a distinct combination of UES-CMA-ES settings (restart range, alpha, gamma, etc.).
- Evaluate multiple fixed policies that switch among these parameter combos.

### Outcome
- Clear differences in final solutions across policies, generally bigger than in Experiments 4 or 5.
- Shows that multi-parameter control leads to significantly more variability and improvement potential.

## Experiment 7

**Goal:** Determine which parameter combos in FinalCombo are the most effective overall.

### Description
- Tests a larger set of possible parameter combinations (24 total).
- Ranks them based on average performance across the benchmark.

### Outcome
- Identifies the top 12 combos that consistently yield good results.
- These 12 combos will then form the action set in the subsequent DRL training experiments.

## Experiment 8

**Goal:** Train a DQN agent in the range-focused environment (FinalRange).

### Description
- Uses the range-based environment with the new, more detailed observation scheme (distances, follower updates, etc.).
- Full DRL training over a subset of benchmark functions.

### Outcome
- The agent significantly improves its rewards during training.
- Confirms that controlling the restart range can be successfully learned by the DQN approach.

## Experiment 9

**Goal:** Train a DQN agent in the multi-parameter environment (FinalCombo) with the selected combos from Experiment 7.

### Description
- Similar DRL pipeline as Experiment 8, but the action space is 12 combos.
- Evaluate how quickly the agent learns vs. the previous single-parameter environment.

### Outcome
- Faster or larger reward improvements than in FinalRange, demonstrating the value of multi-parameter control.
- Sets the stage for further comparisons with standard vs. normalized vs. stagnation-aware reward.

## Experiment 10–16

For brevity, the subsequent experiments explore different reward schemes (standard, normalized, stagnation-aware), fine-tuning of parameter combos, and large-scale evaluations:

- **Experiment 10:** DQN Agent + FinalCombo using standard reward across CEC'13 functions.
- **Experiment 11:** FinalCombo with normalized reward to unify scales across functions.
- **Experiment 12:** FinalCombo with a stagnation-aware penalty in the reward function.
- **Experiment 13–15:** Evaluating trained models with different reward settings on the full benchmark.
- **Experiment 16:** Scaling the DRL hybrid to larger function evaluation budgets (e.g., 900k FEs).

## Experiment 17

**Goal:** Evaluate the DRL-enhanced UES-CMA-ES hybrid on high-dimensional problems (100D).

### Description
- Experiments conducted on the CEC'13 benchmark in 100 dimensions.
- Total budget of 1,000,000 function evaluations.
- Compares the DRL-controlled UES-CMA-ES against UES-CMA-ES, UES, and CMA-ES.
- Uses a trained agent from the FinalCombo environment.

### Outcome
- The DRL hybrid maintains competitive or superior performance in 100D.
- Performance improvements become more pronounced with larger evaluation budgets.
- Confirms scalability of learned restart strategies to higher-dimensional problems.

## Experiment 18

**Goal:** Perform a preliminary sensitivity analysis of the stagnation-aware reward parameters τ (tau) and γ (gamma).

### Description
- Experiments conducted in 30 dimensions with a reduced budget of 30,000 function evaluations.
- Training performed on a small representative subset of CEC'13 functions.
- Multiple values of τ and γ are evaluated to study their interaction with the stagnation-aware reward.
- Trained agents are provided in the `policy/` directory.

### Outcome
- Results indicate that different τ–γ combinations lead to clearly different behaviors.
- Confirms that stagnation sensitivity significantly affects restart decisions.
- Intended as an exploratory analysis rather than a full hyperparameter tuning.

## References

For conceptual details about each environment, parameter combos, and reward function, please see our paper (currently under review).

Specific environment classes are also documented in the code.

For questions or issues, feel free to open an issue in this GitHub or contact the authors.
