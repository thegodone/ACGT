import os
import sys
import time

def get_config(prop):
    tox21_list = [
        'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 
        'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
    ]

    num_epoches, decay_steps = None, None
    if prop in ['bace_c', 'BBBP']:
        num_epoches, decay_steps = 200, 2000
    elif prop in 'HIV':
        num_epoches, decay_steps = 50, 5000
    elif prop in tox21_list:
        num_epoches, decay_steps = 100, 2500
    return num_epoches, decay_steps

def get_error_terminated_jobs(
        prop_list, task_list, seed_list, save_dir,
        prefix, node_dim, graph_dim, dropout_rate, weight_decay    
    ):    

    resubmit_list = []
    for prop in prop_list:
        for task in task_list:
            for seed in seed_list:

                use_attn = task[0]
                use_ln = task[1]
                use_ffnn = task[2]
                concat_readout = task[3]
                readout_method = task[4]
                num_epoches, decay_steps = get_config(prop)

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
                file_name = job_name+'.out' 

                os.system('grep \'Save the predictions\' ' + save_dir+job_name+'.out > tmp.txt')
                f = open('tmp.txt', 'r')
                line = f.readlines()
                if len(line) == 0:
                    print (prop, seed, node_dim, graph_dim, use_attn, use_ln, use_ffnn, \
                           concat_readout, readout_method, dropout_rate, weight_decay)
                    resubmit_list.append(
                        [prop, seed, node_dim, graph_dim, use_attn, use_ln, use_ffnn, 
                         concat_readout, readout_method, dropout_rate, weight_decay]
                    )
    return resubmit_list

save_dir = './results/exp2/'
horus_id = {
    1111:'horus6',
    2222:'horus7',
    3333:'horus8',
    4444:'horus9',
    5555:'horus10',
}

prop_list = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 
    'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
]
prop_list = [
    'bace_c', 'BBBP', 'HIV'
]

# use_attn, use_ln, use_ffnn, use_concat, readout_method
task_list = [
    (False, True, False, True, 'pma'),
    (False, True, True, True, 'pma'),
    (True, True, False, True, 'pma'),
    (True, True, True, True, 'pma')
]

seed_list = [1111, 2222, 3333, 4444, 5555]
prefix = 'ACGT'

node_dim = 64
graph_dim = 128
dropout_rate = 0.1
weight_decay = 1e-6

resubmit_list = get_error_terminated_jobs(
    prop_list, task_list, seed_list, save_dir,
    prefix, node_dim, graph_dim, dropout_rate, weight_decay   
)
exit(-1)
count = 0
for resubmit in resubmit_list:
    count += 1

    job_name = prefix
    for c in resubmit:
        job_name += '_' + str(c)
    file_name = job_name+'.out' 

    prop = resubmit[0]                     
    seed = resubmit[1]                     
    node_dim = resubmit[2]  
    graph_dim = resubmit[3]  
    use_attn = resubmit[4]  
    use_ln = resubmit[5]  
    use_ffnn = resubmit[6]  
    concat_readout = resubmit[7]  
    readout_method = resubmit[8]  
    dropout_rate = resubmit[9]  
    weight_decay = resubmit[10]  
    num_epoches, decay_steps = get_config(prop)

    f=open('test-batch'+str(count)+'.sh','w')
    f.write('''#!/bin/bash
#PBS -N SR_'''+job_name+'''
#PBS -l nodes='''+horus_id[seed]+''':ppn=7
#PBS -l walltime=500:00:00

cd $PBS_O_WORKDIR
echo `cat $PBS_NODEFILE`
cat $PBS_NODEFILE
NPROCS=`wc -l < $PBS_NODEFILE`

for DEVICEID in `seq 0 3`; do
    AVAILABLE=`nvidia-smi -i ${DEVICEID} | grep "No" | wc -l`
    if [ ${AVAILABLE} == 1 ] ; then
        break;
    fi

done
date
source ~/.bashrc
source activate tf-2.0
echo $DEVICEID
export CUDA_VISIBLE_DEVICES=$DEVICEID
export OMP_NUM_THREADS=1

python -u train.py --prefix=''' + prefix + \
                 ' --prop=' + prop + \
                 ' --seed=' + str(seed) + \
                 ' --node_dim=' + str(node_dim) + \
                 ' --graph_dim=' + str(graph_dim) + \
                 ' --use_attn=' + str(use_attn) + \
                 ' --use_ln=' + str(use_ln) + \
                 ' --use_ffnn=' + str(use_ffnn) + \
                 ' --dropout_rate=' + str(dropout_rate) + \
                 ' --weight_decay=' + str(weight_decay) + \
                 ' --readout_method=' + str(readout_method) + \
                 ' --concat_readout=' + str(concat_readout) + \
                 ' --num_epoches=' + str(num_epoches) + \
                 ' --decay_steps=' + str(decay_steps) + \
                 ' > ' + save_dir + file_name + '''

date''')
    f.close()
    os.system('qsub test-batch'+str(count)+'.sh')
    print (job_name, horus_id[seed])
    time.sleep(20.0)
