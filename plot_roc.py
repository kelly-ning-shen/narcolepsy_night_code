import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

savepic = 1
names = ['0.5 min', '1 min','2.5 min', '5 min','10 min', '15 min','30 min', '60 min', '90 min']
files = [
    'multicnnc2cm_0.5min_zscore_shuffle_ROC_onephase',
    'multicnnc2cm_1min_zscore_shuffle_ROC_onephase',
    'multicnnc2cm_2.5min_zscore_shuffle_ROC_onephase1',
    'multicnnc2cm_5min_zscore_shuffle_ROC_onephase',
    'multicnnc2cm_10min_zscore_shuffle_ROC_onephase',
    'multicnnc2cm_15min_zscore_shuffle_ROC_onephase',
    'multicnnc2cm_30min_zscore_shuffle_ROC_onephase',
    'multicnnc2cm_60min_zscore_shuffle_ROC_onephase',
    'multicnnc2cm_90min_zscore_shuffle_ROC_onephase'
]
# colors = [ # colorful
#     '#ffb0b0',
#     '#ffc895',
#     '#000000',
#     '#c9d7b7',
#     '#a3ccba',
#     '#7ec1bd',
#     '#58b6c0',
#     '#3497ba',
#     '#46a0c0'
# ]
colors = [ # deeper
    '#ff7373',
    '#ff8d4d',
    '#000000',
    '#ffa626',
    '#ffc000',
    '#cdb52f',
    '#9aaa5d',
    '#679f8c',
    '#3494ba'
]
def multi_roc(names, files, colors, title='ROC Curve', savepic=0, picpath='Default_ROCcurve.png'):
    '''
    Multiple ROC curve on one plot.
    Args:
        names: list, list of model names
        # ys: ds_subject (y_labels, y_pred_probas on all subjects)
        colors: list of color names
    Return:
        plt
    '''

    plt.subplots(figsize=(5.6,5.2),dpi=220)
    plt.plot([0,1],[0,1],c='slategrey',linestyle='--',zorder=1)
    for (name, file, color) in zip(names, files, colors):
        filepath = f'diagnosis\{file}\ds_subject.txt'
        ds_subject = np.loadtxt(filepath)
        y_labels = ds_subject[:,1]
        y_pred_probas = ds_subject[:,0]
        fpr, tpr, thresholds = roc_curve(y_labels,y_pred_probas,pos_label=1)
        if name == '2.5 min':
            zorder = 3
            plt.fill_between(fpr, tpr, color=color, alpha=0.05,zorder=1)
        else:
            zorder = 2
        plt.plot(fpr, tpr, c=color, label='{} (AUC={:.3f})'.format(name, auc(fpr, tpr)), lw=2,zorder=zorder)
    plt.ylabel('Sensitivity')
    plt.xlabel('1-specificity')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend(loc=4)
    plt.title(title)
    if savepic:
        plt.savefig(picpath)
    plt.close()

if __name__ == '__main__':
    # names = ['multitask+muilticnn', 'one-phase+multicnn', 'multitask+square', 'one-phase+sqaure']
    # files = [
    #     'multicnnc2cm_2.5min_zscore_shuffle_ROC',
    #     'multicnnc2cm_2.5min_zscore_shuffle_ROC_onephase1',
    #     'squaresmall_2.5min_zscore_shuffle_ROC',
    #     'squaresmall_2.5min_zscore_shuffle_ROC_onephase1'
    # ]
    # colors = ['#ff7373',
    # '#ff8d4d',
    # '#ffa626',
    # '#ffc000']
    multi_roc(names, files, colors, title='ROC Curve for NT1 Detection\n(One-Phase + MultiCNNC2CM)', savepic=savepic, picpath='ROC Curve (One-Phase + MultiCNNC2CM).png')