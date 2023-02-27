import numpy as np

BATCH_SIZE = 2
nepoch = 5
ss_outputs = np.arange(50).reshape(BATCH_SIZE,5,nepoch)
ss_dis = ss_outputs.transpose(0,2,1)
ss_dis = ss_dis.reshape(BATCH_SIZE*nepoch,5)
print(ss_dis)