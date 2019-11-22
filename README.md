# ACGT
Accelerating Chemical discovery with Graph Transformer

## TODO
* Firstly, implement the metric calculating "Expected Calibration Error (ECE)"

### Exp 1. Optimizing model architecture
1) Comparison on node embedding methods
2) Comparison on graph embedding methods

|                |           | BACE     |       |       | BBBP     |       |       | HIV      |       |       |
|----------------|-----------|----------|-------|-------|----------|-------|-------|----------|-------|-------|
| Node update    | Readout   | Accuracy | AUROC | AUPRC | Accuracy | AUROC | AUPRC | Accuracy | AUROC | AUPRC |
|    GraphConv   |    Mean   | 0.799    | 0.875 | 0.830 | **0.897**    | 0.909 | **0.965** | 0.967    | 0.726 | 0.230 |
| GraphConv+FFNN |           | 0.787    | 0.884 | 0.844 | 0.880    | 0.899 | 0.956 | 0.967    | 0.724 | 0.233 |
|    GraphAttn   |           | 0.799    | 0.891 | 0.851 | 0.879    | 0.894 | 0.955 | 0.968    | 0.747 | 0.272 |
| GraphAttn+FFNN |           | 0.795    | 0.878 | **0.883** | 0.881    | 0.894 | 0.954 | 0.967    | 0.722 | 0.238 |
|    GraphConv   | Attention | **0.834**    | **0.898** | 0.859 | 0.890    | **0.913** | 0.963 | 0.968    | 0.783 | 0.325 |
| GraphConv+FFNN |           | 0.820    | 0.891 | 0.859 | 0.889    | 0.896 | 0.953 | 0.968    | 0.764 | 0.303 |
|    GraphAttn   |           | 0.815    | 0.891 | 0.859 | 0.879    | 0.889 | 0.949 | 0.969    | **0.790** | **0.370** |
| GraphAttn+FFNN |           | 0.811    | 0.883 | 0.835 | 0.879    | 0.897 | 0.956 | 0.969    | 0.785 | 0.333 |

### Exp 2. Loss for imbalanced data
1) Standard binary crossentropy
2) Oversampling
3) Focal loss
4) Class-balanaced loss
5) Max-margin loss

### Exp 3. Bayesian inference
1) Bayesian inference with MC-dropout
2) Active Learning

### Exp 4. Evaluating model robustness
1) Calibration curve
2) Expected (Maximum) Calibration Error

### Exp 5. Post-hoc interpretation


### Exp 6. Learning with unlabeled data
