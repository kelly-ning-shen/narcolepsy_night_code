from sklearn.metrics import confusion_matrix
import numpy as np
x = np.array([0,1,2,3,1,2])
y = np.array([0,2,2,3,1,3])
classnum = 5
cm = confusion_matrix(x, y)
print(cm)

def cm_uniform(x, y, classnum):
    x1 = np.unique(x)
    y1 = np.unique(y)
    z1 = np.union1d(x1, y1)
    print(z1)
    forgetidx = np.setdiff1d(np.arange(classnum),z1)
    print(forgetidx)
    x = np.concatenate((x, forgetidx), axis=None)
    y = np.concatenate((y, forgetidx), axis=None)
    cm = confusion_matrix(x, y)
    for idx in forgetidx:
        cm[idx, idx] = 0
    return cm

cm = cm_uniform(x, y, classnum)
print(cm)