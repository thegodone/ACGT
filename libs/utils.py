import os

import tensorflow as tf

from libs.lr_scheduler import WarmUpSchedule

def set_cuda_visible_device(ngpus):
    empty = []
    for i in range(4):
        os.system('nvidia-smi -i '+str(i)+' | grep "No running" | wc -l > empty_gpu_check')
        f = open('empty_gpu_check')
        out = int(f.read())
        if int(out)==1:
            empty.append(i)
    if len(empty)<ngpus:
        print ('avaliable gpus are less than required')
    cmd = ''
    for i in range(ngpus):        
        cmd+=str(empty[i])+','
    return cmd

def get_learning_rate_scheduler(lr_schedule='stair', 
                                graph_dim=256, 
                                warmup_steps=1000, 
                                init_lr=1e-3, 
                                decay_steps=500, 
                                decay_rate=0.1, 
                                staircase=True):

    scheduler = None
    if lr_schedule == 'warmup':
        scheduler = WarmUpSchedule(
            d_model=graph_dim, 
            warmup_steps=warmup_steps
        )

    elif lr_schedule == 'stair':

        scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=init_lr,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=staircase
        )            

    return scheduler
