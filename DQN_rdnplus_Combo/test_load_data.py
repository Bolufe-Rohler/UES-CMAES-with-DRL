import csv
import os
import numpy


func_num = 11
dim = 30
experiment = "DQN_rdnplus_ComboX2"
loss_file = f"{experiment}_loss(f{func_num}).csv"
results_file_reward = f"{experiment}_returns(f{func_num}).csv"
results_file_fitness = f"{experiment}_fitness(f{func_num}).csv"


returns = []
fitness = []
loss = []
if os.path.exists(results_file_reward):
    # load existing data
    file = open(loss_file, "r")
    loss.extend(list(csv.reader(file, delimiter=",", quoting=csv.QUOTE_NONNUMERIC)))
    loss = [item for sublist in loss for item in sublist]
    file.close()

    file = open(results_file_reward, "r")
    returns.extend(list(csv.reader(file, delimiter=",", quoting=csv.QUOTE_NONNUMERIC)))
    returns = [item for sublist in returns for item in sublist]
    file.close()

    file = open(results_file_fitness, "r")
    fitness.extend(list(csv.reader(file, delimiter=",", quoting=csv.QUOTE_NONNUMERIC)))
    fitness = [item for sublist in fitness for item in sublist]
    file.close()

print(loss)
print(returns)
print(fitness)

returns.append(666.8)

numpy.savetxt(results_file_reward, returns, delimiter=", ", fmt='% s')
numpy.savetxt(results_file_fitness, fitness, delimiter=", ", fmt='% s')