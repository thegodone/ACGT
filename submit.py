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
        num_epoches, decay_steps = 200, 80
    elif prop in ['tpsa', 'sas', 'logp']:
        num_epoches, decay_steps = 50, 20
    elif prop in 'HIV':
        num_epoches, decay_steps = 100, 40
    elif prop in ['egfr_dude', 'vgfr2_dude', 'tgfr1_dude', 'abl1_dude']:
        num_epoches, decay_steps = 100, 40
    elif prop in tox21_list:
        num_epoches, decay_steps = 100, 40
    return num_epoches, decay_steps

save_dir = './results/exp_screening/'
horus_id = {
    0:'horus11',
    1:'horus10',
    2:'horus12',
    3:'horus13',
}

prop_list = [
    'bace_c', 'BBBP', 'HIV',
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 
    'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
]

prop_list = [
    'logp',
]

prop_list = [
    'bace_c', 'BBBP', 'HIV',
]

prop_list = [
    'egfr_dude', 'tgfr1_dude', 'abl1_dude', 'vgfr2_dude'
]

# use_attn, use_ffnn, use_concat, readout_method
task_list = [
    #(False, False, True, 'mean'),
    #(True, False, True, 'mean'),
    #(False, False, True, 'sum'),
    #(True, False, True, 'sum'),
    (False, False, True, 'pma'),
    #(True, False, True, 'pma'),
]

seed_list = [1111, 2222, 3333, 4444, 5555]
seed_list = [1111]
prefix = 'Stock'
prefix = 'DUDE'

node_dim = 64
graph_dim = 256
dropout_rate = 0.2
prior_length = 1e-4
loss_type = 'bce'
num_layers_list = [4]

count = -1

for prop in prop_list:
    for num_layers in num_layers_list:
        for task in task_list:
            for seed in seed_list:
                count += 1

                use_attn = task[0]
                use_ffnn = task[1]
                concat_readout = task[2]
                readout_method = task[3]
                num_epoches, decay_steps = get_config(prop)

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
                file_name = job_name+'.out' 

                key = count % len(horus_id)

                f=open('test-batch'+str(count)+'.sh','w')
                f.write('''#!/bin/bash
#PBS -N SR_'''+job_name+'''
#PBS -l nodes='''+horus_id[key]+''':ppn=7
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
                 ' --num_layers=' + str(num_layers) + \
                 ' --graph_dim=' + str(graph_dim) + \
                 ' --use_attn=' + str(use_attn) + \
                 ' --use_ffnn=' + str(use_ffnn) + \
                 ' --dropout_rate=' + str(dropout_rate) + \
                 ' --prior_length=' + str(prior_length) + \
                 ' --readout_method=' + str(readout_method) + \
                 ' --concat_readout=' + str(concat_readout) + \
                 ' --num_epoches=' + str(num_epoches) + \
                 ' --decay_steps=' + str(decay_steps) + \
                 ' --loss_type=' + str(loss_type) + \
                 ' --save_model=' + str(True) + \
                 ' > ' + save_dir + file_name + '''

date''')
                f.close()
                os.system('qsub test-batch'+str(count)+'.sh')
                print (job_name, horus_id[key])
                time.sleep(10.0)
