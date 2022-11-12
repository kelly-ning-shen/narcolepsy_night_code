'''
# 根据混淆矩阵，绘制混淆矩阵图，并计算出各类的评价指标和整体准确率 #
    confusion matrix with num
    用于4 实验结果与分析

    注：需要自行保存图片
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from collections import Counter
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix


#confusion matrix
def plot_confusion_matrix(cm, classes_num,mode,savepic=0,picpath='',
                          title='Confusion matrix',
                          cmap='gray_r'):
    '''
    根据混淆矩阵，绘制混淆矩阵图，并计算出各类的评价指标和整体准确率
    '''
    cm_num = cm
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(6.2,5.2))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    title = 'Confusion matrix ('+mode+')'
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(classes_num)
    plt.xticks(tick_marks,['Wake','NREM','REM'],rotation=45)
    plt.yticks(tick_marks,['Wake','NREM','REM'])
    thresh = np.nanmax(cm) / 2 # ignore NaN !
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        # plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
        plt.text(j, i, cm_num[i, j], horizontalalignment="center", # confusion matrix with num
                 color="white" if cm[i, j] > thresh else "black")
        # if cm[i,j] > thresh:
        #     print(i, j, 'white')
        # else:
        #     print(i, j, 'black')

    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    ## 模型分类评价指标
    print('\n======',mode,'======')
    class_metrics,total_acc,avg_sen,avg_spec,marco_f1 = metric(cm_num) # classification metric
    print(class_metrics)
    print('Total accuracy: %.2f%%' % total_acc)
    print('Average sen: %.2f%%' % avg_sen)
    print('Average spec: %.2f%%' % avg_spec)
    print('Macro f1-score: %.2f%%' % marco_f1)
    # plt.savefig('test.png', dpi=200)
    if savepic:
        plt.savefig(picpath)
    # plt.show()
    plt.close()

def metric(cm):
    '''
    cm: confusion_matrix, for example, cm = np.array([[2455,104,306,715],[46,0,11,25],[128,43,331,100],[5,0,14,201]])
    
    Example:
        The f1-score of  1  has ZeroDivisionError.

            acc %  sen %  spec %  ppr %  f1-score
        0  70.92  68.58   80.20  93.20     79.02
        1  94.89   0.00   96.66   0.00      0.00
        2  86.57  54.98   91.47  50.00     52.37
        3  80.84  91.36   80.30  19.31     31.88
    '''
    sum_all = np.sum(cm) # cm中所有元素之和
    sum_col = np.sum(cm,axis=0) # cm中每一列中所有元素之和
    sum_row = np.sum(cm,axis=1) # cm中每一行中所有元素之和
    metrics = {
        'acc %':[],
        'sen %':[],
        'spec %':[],
        'ppr %':[],
        'f1-score':[]
    }
    gross_pos = 0
    for i in range(len(cm)):
        TP = cm[i,i]
        FN = sum_row[i] - TP
        FP = sum_col[i] - TP
        TN = sum_all - TP - FN - FP

        acc = 100*(TP+TN)/sum_all
        sen = 100*TP/(TP+FN)
        spec = 100*TN/(TN+FP)

        gross_pos += TP

        if (FP+TP) != 0:
            ppr = 100*TP/(FP+TP)
        else:
            ppr = 0
            print('\nThe ppr of ',i,' has ZeroDivisionError.\n')
        if (sen+ppr) != 0:
            f1score = 2*sen*ppr/(sen+ppr)
        else:
            f1score = 0
            print('\nThe f1-score of ',i,' has ZeroDivisionError.\n')
            
        metrics['acc %'].append(acc)
        metrics['sen %'].append(sen)
        metrics['spec %'].append(spec)
        metrics['ppr %'].append(ppr)
        metrics['f1-score'].append(f1score)

    pd.set_option('precision',2)
    df = pd.DataFrame(metrics) # DataFrame下的metrics

    total_acc = 100*gross_pos/sum_all
    avg_sen = np.nanmean(metrics['sen %'])
    avg_spec = np.nanmean(metrics['spec %'])
    marco_f1 = np.nanmean(metrics['f1-score'])
    return df,total_acc,avg_sen,avg_spec,marco_f1

# 需要给出验证集和测试集上的混淆矩阵: cm_val, cm_test
#  其实可以根据模型直接得到混淆矩阵图，但是需要相应的数据。
#  目前仅保存了测试集上的数据，但是每次实验中随机划分验证集，没有保存验证集。
#  所以就统一需要自己给出混淆矩阵。这里给出了是 本文模型 的例子。

# cm_val = np.array([[2943,18,3],[20,119,4],[4,4,161]])
# cm_test = np.array([[8167,745,1297],[115,29,3],[90,36,939]])

# plot_confusion_matrix(cm_val,[0,1,2],'validation')
# plot_confusion_matrix(cm_test,[0,1,2],'test')
# plt.show()
# model = load_model('model/right/third_attempt/beat/scale_RRI_NSV/weights_best_beat_B_scale_RRI_NSV_45-0.98.hdf5')
# plot_confuse(model,[test_signals,test_RRIs],test_onehot,'test')
# plt.show()