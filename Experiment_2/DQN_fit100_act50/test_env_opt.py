from env_fit100_act50 import Env_Fit100_Act100
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
import numpy as np
func_num=18
get_new_action = np.array(64, dtype=np.int32)

eval_py_env = Env_Fit100_Act100(func_num, dim=10)
environment = tf_py_environment.TFPyEnvironment(eval_py_env)
time_step = environment.reset()
print(time_step)
cumulative_reward = time_step.reward

for _ in range(10):
    time_step = environment.step(get_new_action)
    print(time_step)
    cumulative_reward += time_step.reward

env = environment.pyenv
print(environment.pyenv.envs[0]._best_fitness)
print(time_step)
cumulative_reward += time_step.reward
print('Final Reward = ', cumulative_reward)
