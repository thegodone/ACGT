import os
import sys

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
    max_val = np.max(pred)
    pred /= max_val
    return label, pred


def top_k_precision(label, pred, k):

    def get_sorted_idx(pred, k):
        sorted_idx = sorted(range(pred.shape[0]), key=lambda s:pred[s])
        return sorted_idx[-k:]

    
    n_pos = np.sum(np.around(pred).astype(int))
    n_key = int(n_pos*k)

    sorted_idx = get_sorted_idx(pred, n_key)

    label = label[sorted_idx]
    pred = np.around(pred[sorted_idx]).astype(int)
    precision = precision_score(label, pred)
    print ("Precision at top", k, ":", precision, n_key)
    return precision


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
    #print ("TP:", tp, "\t FP:", fp, "\t TN :", tn, "\t FN:", fn)

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
        calibration_curve(label, pred, n_bins=bins, strategy='uniform')
    ax1.plot(mean_predicted_value, fraction_of_positives, "s-")
    ax2.hist(pred, range=(0,1), bins=bins, histtype="step", lw=2)

    ece = np.abs(mean_predicted_value - fraction_of_positives).sum()
    #print ("Expected calibration error:", ece)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.set_title("Calibration plots (reliability curve)")

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")

    plt.tight_layout()
    plt.savefig("./figures/"+model_name+"_calibration_curve.png")
    return ece


def reliability_diagram(label, pred, model_name, bins=10):

    width = 1.0 / bins
    bin_centers = np.linspace(0, 1.0-width, bins) + width/2

    conf_bin_mean = np.empty(bins)
    acc_bin_mean = np.empty(bins)
    ece = 0.0
    for i, threshold in enumerate(bin_centers):
        bin_idx = np.logical_and(threshold - width/2 < pred,
                                 pred <= threshold + width)
        conf_bin_mean[i] = pred[bin_idx].mean()
        acc_bin_mean[i] = label[bin_idx].mean()
    
    ece = np.abs(conf_bin_mean - acc_bin_mean).sum()
    print ("Expected Calibration Error:", ece)
    plt.figure()
    plt.plot([0.0, 1.0], 'k:', label="Perfect calibration")
    plt.plot(conf_bin_mean, acc_bin_mean, "s-")
    plt.xlabel("Confidence (output probability)")
    plt.ylabel("Accuracy")
    plt.ylim([-0.05, 1.05])
    #ax1.legend()
    plt.title("Calibration plots (reliability curve)")
    plt.tight_layout()
    plt.savefig("./figures/"+model_name+"_reliability_curve.png")
    return    

def performances(y_truth, y_pred):
    auroc = 0.0
    try:
        auroc = roc_auc_score(y_truth, y_pred)
    except:
        auroc = 0.0    

    y_truth = np.around(y_truth)
    y_pred = np.around(y_pred).astype(int)

    accuracy = accuracy_score(y_truth, y_pred)
    precision = precision_score(y_truth, y_pred)
    recall = recall_score(y_truth, y_pred)
    f1_score = 2*(precision*recall)/(precision+recall+1e-5)
    return accuracy, auroc, precision, recall, f1_score

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
    (False, False, False, True, 'pma'),
]


prefix = 'ACGT'
prefix = 'Bayesian'
prefix = 'Bayesian_re2'
prefix = 'Bayesian_re3'
prefix = 'Bayesian_re4'
prop = 'egfr'
prop = tox21_list[0]
prop = 'BBBP'
prop = 'HIV'
prop = 'bace_c'
prop = sys.argv[2]

node_dim = 64
graph_dim = 256
num_layers = 4
num_heads = 4
weight_decay = 1e-6
dropout_rate = 0.5
dropout_rate = 0.0
dropout_rate = float(sys.argv[3])

mc_dropout = False
mc_dropout = True
mc_dropout = sys.argv[1]
print (mc_dropout)

for task in task_list:
    ece_mean = 0.0
    precision1_mean = 0.0
    precision2_mean = 0.0
    precision3_mean = 0.0
    precision4_mean = 0.0

    auroc_mean = 0.0
    for seed in seed_list:

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
        model_name += '_' + str(use_ffnn)
        model_name += '_' + str(dropout_rate)
        model_name += '_' + str(weight_decay)
        model_name += '_' + str(readout_method)
        model_name += '_' + str(concat_readout)
        model_name += '_' + str(mc_dropout)

        label, pred  = load_results(model_name)
        ece = plot_calibration_curve(label, pred, model_name)
        #reliability_diagram(label, pred, model_name)
        distribution_analysis(label, pred, plot_log=True)
        precision1 = top_k_precision(label, pred, 0.1)
        precision2 = top_k_precision(label, pred, 0.2)
        precision3 = top_k_precision(label, pred, 0.5)
        precision4 = top_k_precision(label, pred, 1.0)

        accuracy, auroc, precision0, recall, f1_score = performances(label, pred)
        auroc_mean += auroc
        
        ece_mean += ece
        precision1_mean += precision1
        precision2_mean += precision2
        precision3_mean += precision3
        precision4_mean += precision4
        #top_k_precision(label, pred, 250)
        #top_k_precision(label, pred, 500)
    ece_mean /= len(seed_list)    
    precision1_mean /= len(seed_list)    
    precision2_mean /= len(seed_list)    
    precision3_mean /= len(seed_list)    
    precision4_mean /= len(seed_list)    
    auroc_mean /= len(seed_list)
    print (ece_mean, precision1_mean, precision2_mean, precision3_mean, precision4_mean, auroc_mean)
    #exit(-1)
