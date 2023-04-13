import numpy as np
import matplotlib.pyplot as plt
import torch
# import torchvision
from sklearn import manifold
# from torchvision.models.feature_extraction import create_feature_extractor

nepoch = 5
nsample = 3000

# EEG = np.random.standard_normal(size=(nepoch, nsample))
# EOG = np.random.standard_normal(size=(nepoch, nsample))
# EMG = np.random.standard_normal(size=(nepoch, nsample))

# signal_pic = np.array([EEG, EOG, EMG])

def input_show(signal_pic):
    # plt.subplots()
    signal_pic = signal_pic.transpose(1,2,0) # 第一层是否是指EEG，第二层是否是指EOG，第三层是否是指EMG
    print(f'Shape of signal_pic: {signal_pic.shape}')
    x = [[1,2],[3,4],[5,6]]
    plt.imshow(signal_pic)
    plt.show()
    plt.close()

# model_trunc = create_feature_extractor(model, return_nodes={'conv3': 'semantic_feature'})

conv3_feature = torch.randn(10,256,1,4) # one segment
# conv3_feature1 = torch.squeeze(conv3_feature,0) # torch.Size([256,4])
conv3_feature1 = torch.squeeze(conv3_feature) # torch.Size([256,4])
print(conv3_feature.shape)
print(conv3_feature1.shape)

ts = manifold.TSNE(random_state=0)
y = ts.fit_transform(conv3_feature1)
print(y.shape)
plt.scatter(y[:,0], y[:,1])
plt.show()
plt.close()
def fearturemap_imshow(featuremap):
    '''featuremap: tensor.Size([C,H,W])'''
    nchannel = featuremap.shape[1]
    nrow = np.sqrt(nchannel)
    plt.figure()
    for i in range(1, nchannel+1): # 256个1*4的特征图输出（conv3）但可惜看不出什么
        plt.subplot(nrow, nrow, i)
        plt.imshow(featuremap[i-1], cmap='gray')
        plt.axis('off')
    plt.show()
    plt.close()
