import numpy as np

a = np.arange(30).reshape(2,3,5)
print(a)
print(a.shape)
b = a[0,:,:][np.newaxis,:]
print(b)
print(b.shape)