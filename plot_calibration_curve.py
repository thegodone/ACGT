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


def plot_calibration_curve(label, pred, model_name, bins=8):
    fraction_of_positives, mean_predicted_value = \
        calibration_curve(label, pred, n_bins=bins, strategy='uniform')
    ece = np.abs(mean_predicted_value - fraction_of_positives).sum()
    print ("Expected calibration error:", ece)
    return fraction_of_positives, mean_predicted_value


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
prefix = 'Bayesian_re4'
prefix = 'Imbalance'
prop = 'egfr'
prop = tox21_list[0]
prop = 'BBBP'
prop = 'HIV'
prop = 'bace_c'
prop = sys.argv[1]

node_dim = 64
graph_dim = 256
num_layers = 4
num_heads = 4
weight_decay = 1e-4

loss = 'focal_' + sys.argv[3]

model_list = [(0.0, False, 'focal_0.5_0.0'), (0.2, False, 'focal_0.5_0.0'), (0.2, True, 'focal_0.5_0.0')]

model_list = [(0.0, False, loss+'_1.0'), (0.0, False, loss+'_1.0'), (0.0, True, loss+'_1.0')]
model_list = [(0.0, False, loss+'_2.0'), (0.0, False, loss+'_2.0'), (0.0, True, loss+'_2.0')]
model_list = [(0.0, False, loss+'_5.0'), (0.0, False, loss+'_5.0'), (0.0, True, loss+'_5.0')]
model_list = [(0.0, False, loss+'_0.0'), (0.0, False, loss+'_0.0'), (0.0, True, loss+'_0.0')]
model_list = [(0.0, False, loss+'_0.0'), (0.0, False, loss+'_1.0'), (0.0, True, loss+'_2.0'), (0.0, True, loss+'_5.0')]
model_list = [(0.0, False, loss+'_0.0'), (0.0, False, loss+'_1.0'), (0.0, True, loss+'_2.0'), (0.0, True, loss+'_5.0')]
model_list = [(0.0, False, loss+'_0.0'), (0.0, False, loss+'_1.0'), (0.0, True, loss+'_2.0'), (0.0, True, loss+'_5.0')]


bins = 10
plot_log = int(sys.argv[2])
for seed in seed_list:

    fop_list = []
    mpv_list = []
    pred_list = []
    for model in model_list:

        use_attn = False
        use_ln = False
        use_ffnn = False
        concat_readout = True
        readout_method = 'pma'

        dropout_rate = model[0]
        mc_dropout = model[1]
        loss_type = model[2]

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
        model_name += '_' + str(loss_type)
        model_name += '_' + str(mc_dropout)

        label, pred  = load_results(model_name)
        fop, mpv = plot_calibration_curve(label, pred, model_name, bins)
        fop_list.append(fop)
        mpv_list.append(mpv)
        pred_list.append(pred)

    fig = plt.figure()
    #fig = plt.figure(figsize=(10,10))
    ax1 = plt.subplot2grid((3,1), (0,0), rowspan=2)
    ax2 = plt.subplot2grid((3,1), (2,0))

    label_list = [
        'p=0.0', 'p=0.2, weight average', 'p=0.2, MC sampling'
    ]
    label_list = [
        '(0.5, 0.0)', '(0.5, 1.0)', '(0.5, 2.0)', '(0.5, 5.0)'
    ]
    label_list = [
        '(0.5, 1.0)', '(0.75, 1.0)', '(0.9, 1.0)'
    ]
    label_list = [
        '(0.5, 2.0)', '(0.75, 2.0)', '(0.9, 2.0)'
    ]
    label_list = [
        '(0.5, 5.0)', '(0.75, 5.0)', '(0.9, 5.0)'
    ]
    label_list = [
        '(0.5, 0.0)', '(0.75, 0.0)', '(0.9, 0.0)'
    ]
    label_list = [
        '('+r'$\alpha$='+sys.argv[3]+', '+r'$\gamma$='+'0.0)',
        '('+r'$\alpha$='+sys.argv[3]+', '+r'$\gamma$='+'1.0)',
        '('+r'$\alpha$='+sys.argv[3]+', '+r'$\gamma$='+'2.0)',
        '('+r'$\alpha$='+sys.argv[3]+', '+r'$\gamma$='+'5.0)',
    ]

    color_list = [
        'r', 'g', 'b', 'c'
    ]    
    
    ax1.plot([0,1], [0,1], "k:", label="Perfectly calibrated")
    for i in range(len(model_list)):
        ax1.plot(mpv_list[i], fop_list[i], "s-", label=label_list[i], c=color_list[i])
        ax2.hist(pred_list[i], range=(0,1), bins=bins, histtype="step", lw=1, label=label_list[i], color=color_list[i])

    #ax1.set_xlabel("Output probability", fontsize=15)
    ax1.set_ylabel("Fraction of positives", fontsize=15)
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(fontsize=12)

    ax2.set_xlabel("Output probability", fontsize=15)
    ax2.set_ylabel("Count", fontsize=15)
    if plot_log == 1:
        ax2.set_yscale("log")

    plt.tight_layout()
    plt.savefig("./figures/"+prop+'_'+str(seed)+'_'+sys.argv[3]+"_calibration_curve_re.png")
