# Deep Reinforcement Learning for Multi-Restart Strategies in UES-CMA-ES

This repository contains the code and experiment folders for our work on using deep reinforcement learning (DRL) to control restart decisions in the **UES-CMA-ES** hybrid metaheuristic. The experiments document the progression from early restart-point environments that did not learn effectively to the final restart-level environment in which the RL agent selects restart-related parameter configurations.

The overall research question is whether restart decisions in a hybrid metaheuristic can be made **learnable** by designing the RL environment appropriately. Across the experiments, we study environment design, action-space construction, observation design, reward formulation, and the effect of the specific value-based RL variant.

## Repository overview

Each experiment folder contains some combination of:
- environment definitions
- training scripts
- evaluation scripts
- result files, plots, or logs
- saved policies for trained agents

For experiments that rely on trained reinforcement learning agents, the trained models are typically available in the `policy/` directory inside the corresponding experiment folder. These saved policies can be loaded directly for evaluation and reproducibility without retraining.

## Experimental roadmap

The experiments are organized in roughly four stages:

1. **Early restart-point environments**  
   Experiments 1–3 test environments where the agent chooses among saved restart locations from the current run.

2. **Final-solution restart environments**  
   Experiments 4–7 shift to restarting from the final solution of the previous cycle while varying restart-related parameters.

3. **Learning in FinalRange / FinalCombo**  
   Experiments 8–18 develop the main DQN-based framework, compare reward variants, and evaluate large-scale and high-dimensional behavior.

4. **Revision-driven validation experiments**  
   Experiments 19–21 add targeted studies on observation ablation, reward ablation, and DQN versus Double DQN.

---

## Experiment 1

**Goal:** Test the initial **Fit100-Act100** environment on a small set of functions, where the agent selects one of 100 recorded solutions from the current run as the restart point.

### Description
- The action space consists of 100 restart snapshots sampled at equidistant points during the run.
- No parameter changes are applied; only the restart location differs.

### Outcome
- Training shows little or no meaningful improvement in reward.
- The learned policy does not clearly differentiate among the 100 possible restart points.
- This suggests that restart-point selection alone is not informative enough in this environment.

---

## Experiment 2

**Goal:** Investigate whether reducing the action space helps the initial Fit100-based environment.

### Description
- Variants such as **Fit100-Act50** and **Fit100-Act10** reduce the number of restart-point choices.
- The environment remains otherwise unchanged from Experiment 1.

### Outcome
- Performance remains weak and reward curves still resemble random behavior.
- Simply shrinking the action space does not resolve the limitations of the environment.

---

## Experiment 3

**Goal:** Test whether the failure of the initial environment is due to poor learning or to the environment itself.

### Description
- Uses the same **Fit100-Act100** environment as before.
- Instead of training an RL agent, several fixed hand-crafted policies are evaluated directly.

### Outcome
- Different fixed policies produce nearly identical results.
- This strongly suggests that the environment itself is the problem: the available restart choices do not have enough meaningful impact on the optimization process.

---

## Experiment 4

**Goal:** Introduce the **FinalRange** environment, where restarts are anchored at the final solution of the previous cycle and the action controls the restart range.

### Description
- The action determines how widely new UES solutions are sampled around the current incumbent.
- Different fixed policies are evaluated to test whether this parameter creates meaningful behavioral differences.

### Outcome
- Policies now produce clearly different final errors.
- Restart range control is much more informative than choosing among stored restart points.
- This marks an important shift toward restart-level control that the agent can potentially learn.

---

## Experiment 5

**Goal:** Introduce the **FinalRate** environment, where the action controls the UES convergence-rate parameter at each restart.

### Description
- Restarts remain anchored at the final solution.
- The action selects among discrete convergence-rate settings.

### Outcome
- Policy differences are visible, though typically less pronounced than in FinalRange.
- Convergence-rate control alone is helpful but weaker than restart-range control.

---

## Experiment 6

**Goal:** Introduce the **FinalCombo** environment, where each action corresponds to a combination of restart-related parameter settings.

### Description
- The action space includes combinations of settings such as restart range, convergence behavior, and search-budget allocation.
- Fixed policies are evaluated to determine whether combining parameters creates stronger behavioral differences.

### Outcome
- Policy outcomes differ much more clearly than in Experiments 4 and 5.
- Multi-parameter restart control creates a more expressive and promising RL environment.

---

## Experiment 7

**Goal:** Identify the most effective parameter combinations for FinalCombo.

### Description
- A larger pool of candidate parameter combinations is tested.
- Combinations are ranked based on average performance across benchmark functions.

### Outcome
- The top 12 combinations are selected to define the FinalCombo action space used in later learning experiments.
- This reduces the action space to a tractable but still expressive set of restart configurations.

---

## Experiment 8

**Goal:** Train a DQN agent in the **FinalRange** environment.

### Description
- Uses the range-based action space together with a richer observation representation.
- Training is performed on a subset of benchmark functions.

### Outcome
- Reward improves substantially during training.
- Confirms that restart-range control is learnable by a DQN agent in a properly designed environment.

---

## Experiment 9

**Goal:** Train a DQN agent in the **FinalCombo** environment using the selected parameter combinations from Experiment 7.

### Description
- Uses the same general DRL pipeline as Experiment 8, but with the multi-parameter FinalCombo action space.
- Tests whether the richer action space improves learning quality.

### Outcome
- Learning is stronger than in the earlier environments.
- This experiment establishes FinalCombo as the main environment for subsequent studies.

---

## Experiment 10

**Goal:** Train a DQN agent in the FinalCombo environment using the **standard reward**.

### Description
- Uses the final multi-parameter environment across benchmark functions.
- Serves as the baseline DQN training configuration for later reward comparisons.

### Outcome
- Produces competitive learned restart policies.
- Serves as the reference point for subsequent normalized and stagnation-aware reward variants.

---

## Experiment 11

**Goal:** Train FinalCombo using a **normalized reward**.

### Description
- Modifies the reward to reduce scale differences across benchmark functions.
- Keeps the main environment and DQN training structure unchanged.

### Outcome
- Improves cross-function comparability of rewards.
- Helps assess whether reward-scale normalization benefits benchmark-wide training.

---

## Experiment 12

**Goal:** Train FinalCombo using a **stagnation-aware reward**.

### Description
- Extends the reward formulation to penalize repeated non-improving restarts.
- Designed to encourage more adaptive behavior when the search stagnates.

### Outcome
- Produces more robust learning in many cases.
- Motivates later sensitivity and ablation experiments around reward design.

---

## Experiment 13

**Goal:** Evaluate trained FinalCombo agents obtained with the standard reward.

### Description
- Loads saved policies and evaluates them on the benchmark suite.
- Focuses on the optimization quality of the trained restart policy rather than training curves.

### Outcome
- Provides benchmark-level evidence for the effectiveness of the standard-reward DQN agent.

---

## Experiment 14

**Goal:** Evaluate trained FinalCombo agents obtained with the normalized reward.

### Description
- Uses the same evaluation protocol as Experiment 13.
- Compares the effect of the normalized reward on final optimization performance.

### Outcome
- Shows how reward normalization affects benchmark-level results and robustness.

---

## Experiment 15

**Goal:** Evaluate trained FinalCombo agents obtained with the stagnation-aware reward.

### Description
- Evaluates saved policies trained with the stagnation-aware reward on the full benchmark.
- Allows direct comparison with the standard and normalized reward settings.

### Outcome
- Helps establish whether stagnation-aware reward shaping leads to better final policies.

---

## Experiment 16

**Goal:** Study larger-budget behavior for the DRL-enhanced UES-CMA-ES hybrid.

### Description
- Uses larger function-evaluation budgets than the earlier experiments.
- Examines whether the learned restart policies remain useful as the optimization budget grows.

### Outcome
- Shows that the learned policies remain competitive at larger budgets.
- Supports the practical relevance of offline-trained restart policies.

---

## Experiment 17

**Goal:** Evaluate the DRL-enhanced UES-CMA-ES hybrid on **100-dimensional** problems.

### Description
- Uses the IEEE CEC’13 benchmark in 100 dimensions.
- Compares the DRL-controlled hybrid against UES-CMA-ES, UES, and CMA-ES.
- Reuses a trained agent from the FinalCombo environment.

### Outcome
- The DRL hybrid remains competitive in the higher-dimensional setting.
- Supports the idea that the learned restart policy captures structural cues that transfer beyond the original 30D training setup.

---

## Experiment 18

**Goal:** Perform a preliminary sensitivity analysis of the stagnation-aware reward parameters.

### Description
- Explores different values of the reward parameters controlling stagnation sensitivity.
- Uses a smaller set of functions and a reduced budget to study parameter interaction efficiently.

### Outcome
- Different parameter settings lead to noticeably different behaviors.
- Confirms that the stagnation-aware reward is meaningful and worth studying further.

---

## Experiment 19

**Goal:** Perform an **observation-space ablation** for the FinalCombo environment.

### Description
- Tests whether the full 65-dimensional observation vector is necessary.
- Compares the baseline observation design against reduced variants:
  - **Baseline:** full checkpoint summaries plus single-run features
  - **NoSingle:** removes the 5 single-run features
  - **SS10:** reduces checkpoint resolution from 20 to 10 while keeping the single-run features
- Uses the same FinalCombo environment and standard DQN setting.

### Outcome
- The full observation design performs best.
- Removing the single-run features degrades performance.
- Reducing checkpoint resolution causes a much larger performance drop.
- Supports the final 65-dimensional observation design used in the paper.

---

## Experiment 20

**Goal:** Compare alternative **reward formulations** under a controlled FinalCombo training setting.

### Description
- Compares:
  - **Standard reward**
  - **Normalized reward**
  - **Stagnation-aware reward**
- Uses the same general environment, training protocol, and evaluation structure so that differences can be attributed mainly to reward design.

### Outcome
- The stagnation-aware reward produces the strongest and most consistent overall results.
- The normalized reward often improves over the standard reward, but remains behind the stagnation-aware variant.
- Confirms that reward design materially affects restart-level learning and final optimization quality.

---

## Experiment 21

**Goal:** Compare **DQN** and **Double DQN** in the FinalCombo environment.

### Description
- Trains and evaluates both algorithms under the same:
  - environment
  - observation representation
  - action space
  - reward setting
  - training protocol
- Designed as a robustness check on the choice of value-based RL algorithm.

### Outcome
- Double DQN shows modest improvements in some metrics and functions.
- The overall differences remain moderate.
- Supports the main claim that the effectiveness of the method depends more on the restart-level environment design than on the specific choice between DQN and Double DQN.

---

## Notes on interpretation

A key message of this repository is that the main challenge is not simply to attach reinforcement learning to a metaheuristic, but to design an environment in which restart-level decisions are both meaningful and learnable. The early experiments show that poorly chosen restart actions provide little signal for learning, while the later experiments show that carefully designed restart-parameter environments can support useful policies.

## Related paper

For the conceptual details behind the environments, observation design, reward functions, and benchmark evaluations, please see the associated paper and revision materials.

## Contact

For questions, issues, or suggestions, please open an issue in this repository or contact the authors.