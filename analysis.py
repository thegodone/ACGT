import os

import numpy as np
import matplotlib.pyplot as plt

def extract_results(save_dir, model_name):
    os.system('grep \'Test\' ' + save_dir+model_name+'.out > tmp.txt')
    f = open('tmp.txt', 'r')
    line = f.readlines()[-1].split()
    return [float(line[2*(i+1)]) for i in range(5)]

   
save_dir = './results/exp1/'
task_list = [
    (False, True, False, True, 'linear'),
    (False, True, True, True, 'linear'),
    (True, True, False, True, 'linear'),
    (True, True, True, True, 'linear'),
    (False, True, False, True, 'pma'),
    (False, True, True, True, 'pma'),
    (True, True, False, True, 'pma'),
    (True, True, True, True, 'pma'),
]

seed_list = [1111, 2222, 3333, 4444, 5555]

tox21_list = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 
    'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
]

prefix = 'ACGT'
prop = 'bace_c'
prop = 'BBBP'
prop = 'HIV'
prop = tox21_list[10]
print (prop)

node_dim = 64
graph_dim = 128
dropout_rate = 0.0
weight_decay = 0.0

count = 0
for task in task_list:
    results_list = []
    for seed in seed_list:
        count += 1

        use_attn = task[0]
        use_ln = task[1]
        use_ffnn = task[2]
        concat_readout = task[3]
        readout_method = task[4]

        job_name = prefix
        job_name += '_' + prop
        job_name += '_' + str(seed)
        job_name += '_' + str(node_dim)
        job_name += '_' + str(graph_dim)
        job_name += '_' + str(task[0])
        job_name += '_' + str(task[1])
        job_name += '_' + str(task[2])
        job_name += '_' + str(task[3])
        job_name += '_' + str(task[4])
        job_name += '_' + str(dropout_rate)
        job_name += '_' + str(weight_decay)

        results = extract_results(save_dir, job_name)
        results_list.append(results)
    avg, std = np.mean(results_list, axis=0), np.std(results_list, axis=0)
    print (avg)
