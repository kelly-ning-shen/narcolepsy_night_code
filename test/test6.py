import numpy as np
import matplotlib.pyplot as plt

# float array, 3 channels
# For float array with 3 channels, by  default the values out of range [0,1] are **Clipped** !

x = np.ones([300,300,3])
# x *= 0.4
# for i in range(100,200):
#     x[i] = np.ones([600,3])*2.5
# for i in range(250,300):
#     x[i] = np.ones([600,3])*-2
# for i in range(350,400):
#     x[i] = np.ones([600,3])*0.7

R = np.ones([300,300])
G = np.ones([300,300])
B = np.ones([300,300])

R[0:50,:] = 0.5
G[0:50,:] = 0
B[0:50,:] = 0

R[100:150,:] = 0
B[100:150,:] = 0

R[200:250,:] = 0
G[200:250,:] = 0

x[:,:,0] = R
x[:,:,1] = G
x[:,:,2] = B
print(x.dtype)
plt.imshow(x)
plt.colorbar()
plt.show()