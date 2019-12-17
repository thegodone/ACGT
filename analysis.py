import os
import sys

import numpy as np
import matplotlib.pyplot as plt

K = 2
K = 5

idx = int(sys.argv[1])

def extract_results(save_dir, model_name):
    os.system('grep \'Test\' ' + save_dir+model_name+'.out > tmp.txt')
    f = open('tmp.txt', 'r')
    lines = f.readlines()
    line = lines[idx].split()
    return [float(line[2*(i+1)]) for i in range(K)], len(lines)

   
save_dir = './results/exp1_re/'
save_dir = './results/exp_bio/'
save_dir = './results/exp_concat/'
save_dir = './results/exp_stock/'
save_dir = './results/exp_bayesian/'
task_list = [
    #(False, False, True, 'mean'),
    #(False, False, True, 'sum'),
    (False, False, True, 'pma'),
    #(True, False, True, 'mean'),
    #(True, False, True, 'sum'),
    #(True, False, True, 'pma'),
]

seed_list = [1111]
seed_list = [1111, 2222, 3333, 4444, 5555]

tox21_list = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 
    'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
]

num_layers_list = [2,3,4,5,6]
num_layers_list = [4]

prefix = 'ACGT'
prefix = 'Bayesian_re'
prefix = 'Bayesian_re2'
prefix = 'Bayesian_re3'
prefix = 'ACGT_no_ln'
prefix = 'Stock'
prefix = 'Bayesian_re4'
prop = tox21_list[10]
prop = 'sas'
prop = 'tpsa'
prop = 'logp'
prop = 'bace_c'
prop = 'BBBP'
prop = 'HIV'
prop = sys.argv[2]
print (prop)

node_dim = 64
graph_dim = 256
dropout_rate = float(sys.argv[3])
#prior_length = 0.1
prior_length = 1e-6

count = 0

plt.figure()
for task in task_list:
    val_list = []
    for num_layers in num_layers_list:
        results_list = []
        for seed in seed_list:
            count += 1

            use_attn = task[0]
            use_ffnn = task[1]
            concat_readout = task[2]
            readout_method = task[3]

            job_name = prefix
            job_name += '_' + prop
            job_name += '_' + str(seed)
            job_name += '_' + str(node_dim)
            job_name += '_' + str(num_layers)
            job_name += '_' + str(graph_dim)
            job_name += '_' + str(task[0])
            job_name += '_' + str(task[1])
            job_name += '_' + str(task[2])
            job_name += '_' + str(task[3])
            job_name += '_' + str(dropout_rate)
            job_name += '_' + str(prior_length)

            results, n = extract_results(save_dir, job_name)
            results_list.append(results)
        avg, std = np.mean(results_list, axis=0), np.std(results_list, axis=0)
        print (avg, n, job_name)
        #print (std, n, job_name)
        val_list.append(avg[1])

    use_attn = task[0]
    readout_method = task[3]

    line = {True: '-o', False: '-v'}
    node = {True: 'GA', False: 'GC'}

    color = {'mean': 'b', 'sum': 'g', 'pma': 'r'}
    agg = {'mean': 'Average', 'sum': 'Sum', 'pma': 'Attention'}

    plt.plot(num_layers_list, val_list, line[use_attn], c=color[readout_method], label=node[use_attn]+', '+agg[readout_method])
plt.xlabel('Number of graph convolution layers', fontsize=15)
plt.ylabel('Root Mean Squared Error', fontsize=15)    
plt.xticks(num_layers_list, fontsize=15)
plt.yticks(fontsize=15)
#plt.yticks([0.0, 0.05, 0.10, 0.15, 0.20, 0.25], fontsize=15)
#plt.yscale('log')
plt.tight_layout()
plt.legend(fontsize=12, loc='best')
plt.savefig('exp1_num_layers.png')
