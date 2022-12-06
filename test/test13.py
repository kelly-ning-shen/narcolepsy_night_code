import numpy as np
f = open('log/TEST_15min_ROC.txt','r',encoding='utf-8').readlines()
d_preds_15min = []
d_labels_15min = []
for i in range(len(f)):
    a = f[i]
    print(type(a))
    d_preds = list(map(float, f[i].split(' ')))
    if i < 23:
        label = 0
    else:
        label = 1
    d_labels = [label]*len(d_preds)
    d_preds_15min += d_preds
    d_labels_15min += d_labels

print(d_preds_15min)
print(d_labels_15min)