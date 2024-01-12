from env_rdnplus_combo import Env_Rdnplus_comboX2
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
import numpy as np

func_num = 12
get_new_action = np.array(64, dtype=np.int32)
dim = 30

eval_py_env = Env_Rdnplus_comboX2(func_num, dim=dim, minimum=-300)
environment = tf_py_environment.TFPyEnvironment(eval_py_env)
time_step = environment.reset()
print(time_step)
cumulative_reward = time_step.reward

for _ in range(10):
    get_new_action = 4 #np.array(np.random.randint(0, 6))  # environment.step(get_new_action)
    print(get_new_action)
    time_step = environment.step(get_new_action)
    print(time_step)
    cumulative_reward += time_step.reward

env = environment.pyenv
print(environment.pyenv.envs[0]._best_fitness)
print(time_step)
cumulative_reward += time_step.reward
print('Final Reward = ', cumulative_reward)
