import os

import numpy as np
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

def load_results(model_name):
    label = np.load('./outputs/'+model_name+'_label.npy')
    pred = np.load('./outputs/'+model_name+'_pred.npy')
    return label, pred

seed_list = [1111, 2222, 3333, 4444, 5555]
tox21_list = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 
    'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
]
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
task_list = [
    (True, True, False, True, 'pma'),
]

def distribution_analysis(label, pred, plot_log):

    label = label.astype(int)
    pred_int  = np.around(pred).astype(int)
    num_tot = pred.shape[0]

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    tp_list = []
    fp_list = []
    tn_list = []
    fn_list = []
    for i in range(num_tot):
        if(label[i] == 0):
            if(pred_int[i] == 0):
                tn += 1
                tn_list.append(i)
            else:
                fp += 1
                fp_list.append(i)
        else:
            if(pred_int[i] == 0):
                fn += 1
                fn_list.append(i)
            else:
                tp += 1
                tp_list.append(i)
    print ("TP:", tp, "\t FP:", fp, "\t TN :", tn, "\t FN:", fn)

    num_bins  = 20
    plt.figure()
    n, bins, patches = plt.hist(np.asarray(pred[tp_list]), num_bins, label='TP', color='tab:blue',   histtype='step', lw=2)
    n, bins, patches = plt.hist(np.asarray(pred[fp_list]), num_bins, label='FP', color='tab:orange', histtype='step', lw=2)
    n, bins, patches = plt.hist(np.asarray(pred[tn_list]), num_bins, label='TN', color='tab:green',  histtype='step', lw=2)
    n, bins, patches = plt.hist(np.asarray(pred[fn_list]), num_bins, label='FN', color='tab:red',    histtype='step', lw=2)
    plt.xlabel('Output probability')
    plt.ylabel('Frequency')
    plt.legend()
    if plot_log:
        plt.yscale('log')
    plt.savefig('./figures/'+model_name+'_prediction_distribution.png')
    return

def plot_calibration_curve(label, pred, model_name, bins=10):

    fig = plt.figure(figsize=(10,10))
    ax1 = plt.subplot2grid((3,1), (0,0), rowspan=2)
    ax2 = plt.subplot2grid((3,1), (2,0))

    ax1.plot([0,1], [0,1], "k:", label="Perfectly calibrated")
    fraction_of_positives, mean_predicted_value = \
        calibration_curve(label, pred, n_bins=bins)
    ax1.plot(mean_predicted_value, fraction_of_positives, "s-")
    ax2.hist(pred, range=(0,1), bins=bins, histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    #ax1.legend()
    ax1.set_title("Calibration plots (reliability curve)")

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    #ax2.legend()

    plt.tight_layout()
    plt.savefig("./figures/"+model_name+"_calibration_curve.png")
    return

prefix = 'ACGT'
prefix = 'GTA'
prop = tox21_list[0]
prop = 'HIV'
prop = 'bace_c'

node_dim = 64
graph_dim = 256
num_layers = 4
num_heads = 4
dropout_rate = 0.0
weight_decay = 1e-4
dropout_rate = 0.1
mc_dropout = False
#for task in task_list:
#    for seed in seed_list:
for seed in seed_list:
    for task in task_list:

        use_attn = task[0]
        use_ln = task[1]
        use_ffnn = task[2]
        concat_readout = task[3]
        readout_method = task[4]

        model_name = prefix
        model_name += '_' + prop
        model_name += '_' + str(seed)
        model_name += '_' + str(num_layers)
        model_name += '_' + str(node_dim)
        model_name += '_' + str(graph_dim)
        model_name += '_' + str(use_attn)
        model_name += '_' + str(num_heads)
        model_name += '_' + str(use_ln)
        model_name += '_' + str(use_ffnn)
        model_name += '_' + str(dropout_rate)
        model_name += '_' + str(weight_decay)
        model_name += '_' + str(readout_method)
        model_name += '_' + str(concat_readout)
        model_name += '_' + str(mc_dropout)

        label, pred  = load_results(model_name)
        plot_calibration_curve(label, pred, model_name)
        distribution_analysis(label, pred, plot_log=False)
    exit(-1)
