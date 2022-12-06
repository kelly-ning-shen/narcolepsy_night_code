'''ROC curve'''
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np

d_labels = np.ones(78)
d_labels[0:23] = 0
d_preds = np.array([0.23614630232063624, 0.2982520750590733, 0.1471606986897607, 0.990499390496148, 0.48786142468452454, 
    0.6632994403099192, 0.27862266331378904, 0.33210736997425555, 0.8203566333939952, 0.10226823227299799, 0.8943385265527233, 
    0.4704191204040281, 0.10885335753361385, 0.3201542053785589, 0.08741158352543911, 0.30727410059550714, 0.9450035156874821, 
    0.35468362718820573, 0.0849589934765265, 0.2941284026536677, 0.4554982584502016, 0.11216800920665264, 0.550341035425663, 
    0.8301968425512314, 0.9513342380523682, 0.9906026973868861, 0.9951260406523943, 0.9354730865289999, 0.7918661336104075, 
    0.43342650505272967, 0.8006111518029244, 0.9977849090800566, 0.9907408431172371, 0.9499442062594674, 0.9920127101846643, 
    0.4672767978274461, 0.9654342116731586, 0.6113037566997503, 0.9714348415533701, 0.8925258156025049, 0.5967864962042989, 
    0.9998811233428216, 0.9049215465784073, 0.8010847247427418, 0.9919474720954895, 0.8948475250176021, 0.6422766557624263, 
    0.9832464141004226, 0.9921419655575472, 0.8362531572580337, 0.23118889734551712, 0.7070458480290004, 0.9548060117345868, 
    0.9386251007809359, 0.97515150478908, 0.8255057172740207, 0.9608516938546124, 0.9835933382446701, 0.9722384827477591, 
    0.9416987415817049, 0.9083254587005924, 0.7745533120818436, 0.9016616974558149, 0.8037866156548261, 0.8772686968247095, 
    0.6976446642957884, 0.13761497584774213, 0.9571335174971156, 0.8181334651178784, 0.26166454589728155, 0.8321580688158671, 
    0.9546887713509637, 0.6651177632848959, 0.5458131904403368, 0.8694714804490408, 0.9900949194150812, 0.9860524438522957, 0.9970984998203459])

def plot_ROC_curve(y_labels, y_pred_probas, mode, savepic=0, picpath='Default_ROCcurve.png'):
    fpr, tpr, thresholds = roc_curve(y_labels,y_pred_probas,pos_label=1)
    auc = roc_auc_score(y_labels,y_pred_probas)
    for i, threshold in enumerate(thresholds):
        print(f'fpr: {fpr[i]}, tpr: {tpr[i]}, threshold: {threshold}, ')
    maxidx = (tpr-fpr).tolist().index(max(tpr-fpr))
    best_threshold = thresholds[maxidx]
    best_fpr = fpr[maxidx]
    best_tpr = tpr[maxidx]
    print(f'\n=== best_threshold: {best_threshold}, best_fpr: {best_fpr}, best_tpr: {best_tpr} ===')

    # create ROC curve
    title = f'ROC curve ({mode})'
    plt.figure(figsize=(5.6,5.2))
    plt.plot([0,1],[0,1],c='slategrey',linestyle='--',zorder=1)
    plt.plot(fpr, tpr, c='goldenrod', label='ROC curve (area = {0:.2f})'.format(auc), lw=2,zorder=2)
    plt.ylabel('True positive rate')
    plt.xlabel('False positive rate')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.legend(loc=4)
    plt.scatter(best_fpr, best_tpr, s=30, c='r',linewidths=1, edgecolors='k',zorder=3)
    
    plt.title(title)
    if savepic:
        plt.savefig(picpath)
    plt.show()
    plt.close()

if __name__ == '__main__':
    plot_ROC_curve(d_labels, d_preds, 'test')
