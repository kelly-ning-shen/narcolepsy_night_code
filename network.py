import torch
import torch.nn as nn

####################### MODEL COMPONENTS #######################
class SpConv2d(nn.Module):
    '''NeurlPS2019: Convolution with even-sized kernels and symmetric padding'''
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,*args,**kwargs):
        super(SpConv2d,self).__init__()
        self.spconv = nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding)
    def forward(self,x):
        n,c,h,w = x.size()
        assert c % 4 == 0
        x1 = x[:,:c//4,:,:]
        x2 = x[:,c//4:c//2,:,:]
        x3 = x[:,c//2:c//4*3,:,:]
        x4 = x[:,c//4*3:c,:,:]
        x1 = nn.functional.pad(x1,(1,0,1,0),mode = "constant",value = 0) # left top
        x2 = nn.functional.pad(x2,(0,1,1,0),mode = "constant",value = 0) # right top
        x3 = nn.functional.pad(x3,(1,0,0,1),mode = "constant",value = 0) # left bottom
        x4 = nn.functional.pad(x4,(0,1,0,1),mode = "constant",value = 0) # right bottom
        x = torch.cat([x1,x2,x3,x4],dim = 1)
        return self.spconv(x)

class DepthwiseConv(nn.Module):
    '''(可以用于第一层卷积层！读取每个通道的信息！)
    Mentioned in: 
    1. Xception
    2. MobileNet'''
    def __init__(self, in_channels, out_channels):
        super(DepthwiseConv, self).__init__()
        self.depth_wise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.point_wise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        x = self.depth_wise(x)
        x = self.point_wise(x)
        return x

class SingleConv(nn.Module):
    '''[kernel: square] (conv -> BN -> ReLU)'''
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(SingleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class OutSleepStage(nn.Module):
    '''conv [kernel: 1*1]*out_channels)'''
    def __init__(self, in_channels, out_channels):
        super(OutSleepStage, self).__init__()
        self.outsleepstage = nn.Conv2d(in_channels,out_channels,kernel_size=1,padding=0)
    def forward(self, x):
        x = self.outsleepstage(x)
        return x

class OutDiagnosis(nn.Module):
    '''(flatten -> fc -> fc)'''
    def __init__(self, in_channels, hidden_channels):
        # in_channels: the number of features that are input to the fully-connected layer after flattening
        super(OutDiagnosis, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, 1)
        )
    def forward(self, x):
        x = self.fc(x)
        return x

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x*y.expand_as(x)

class CNNBlock(nn.Module):
    '''Zhou Wei: A Lightweight Segmented Attention Network for Sleep Staging by Fusing Local Characteristics and Adjacent Information'''
    def __init__(self, in_channels, kernel_size):
        super(CNNBlock, self).__init__()
        self.cnnblock = nn.Sequential(
            SingleConv(in_channels, out_channels=64, kernel_size=kernel_size, stride=(1,2), padding=0),
            nn.MaxPool2d(kernel_size=(1,4), stride=(1,4)),
            nn.Dropout(p=0.5),
            SingleConv(in_channels=64, out_channels=128, kernel_size=(1,3), stride=(1,1), padding=0),
            nn.MaxPool2d(kernel_size=(1,4), stride=(1,4)),
            SingleConv(in_channels=128, out_channels=128, kernel_size=(1,3), stride=(1,1), padding=0),
            nn.MaxPool2d(kernel_size=(1,4), stride=(1,4))
        )
    def forward(self, x):
        x = self.cnnblock(x)
        return x

class MultiScalerCNN(nn.Module):
    '''Zhou Wei: A Lightweight Segmented Attention Network for Sleep Staging by Fusing Local Characteristics and Adjacent Information'''
    def __init__(self, in_channels):
        super(MultiScalerCNN, self).__init__()
        self.cnnblock1 = CNNBlock(in_channels,kernel_size=(1,50))
        self.cnnblock2 = CNNBlock(in_channels,kernel_size=(1,20))
        self.dropout = nn.Dropout(p=0.5)
    def forward(self, x):
        x1 = self.cnnblock1(x)
        x2 = self.cnnblock2(x)
        x = torch.cat((x1, x2), dim=3) # TODO: concat （在哪一个维度？
        x = self.dropout(x)
        return x

class MultiCNN_SE(nn.Module):
    def __init__(self, in_channels):
        super(MultiCNN_SE, self).__init__()
        self.multicnn = MultiScalerCNN(in_channels)
        self.se = SELayer(channel=128)
    def forward(self, x):
        x = self.multicnn(x)
        x = self.se(x)
        return x # torch.Size([10, 128, 30, 179])

####################### MODELS #######################
class MultiCNNC2CM(nn.Module):
    '''
    Plan 2: input: 3*30*3000
    - kernel_shape: C2CM (not square)
    - kernel_size: big, small
    - structure: encoder
    combine: SAN (Wei Zhou), C2CM (Cuntai Guan)
    parameter: nepoch: num of epochs in one xx min duration (for example, nepoch=30 when 15min)
    '''
    def __init__(self, n_channels, nepoch):
        super(MultiCNNC2CM, self).__init__()
        self.multicnn_se = MultiCNN_SE(n_channels)
        self.conv1 = SingleConv(128, 128, kernel_size=(1,7), stride=(1,3), padding=0)
        self.conv2 = SingleConv(128, 128, kernel_size=(1,3), stride=(1,3), padding=0)

        self.conv2_ss1 = SingleConv(128, 5, kernel_size=(1,4), stride=1, padding=0)
        self.out_ss = OutSleepStage(5,5)

        self.conv3 = SingleConv(128, 256, kernel_size=(nepoch,1), stride=1, padding=0)
        self.out_d = OutDiagnosis(1024, hidden_channels=256)
    def forward(self, x):
        x = self.multicnn_se(x)
        x = self.conv1(x)
        x = self.conv2(x)

        ss = self.conv2_ss1(x)
        ss = self.out_ss(ss)
        ss = torch.squeeze(ss, 3)

        d = self.conv3(x)
        d = self.out_d(d)
        return ss, d

class MultiCNNC2CM_S(nn.Module):
    '''
    Plan 2: input: 3*30*3000
    - kernel_shape: C2CM (not square)
    - kernel_size: big, small
    - structure: encoder
    combine: SAN (Wei Zhou), C2CM (Cuntai Guan)
    parameter: nepoch: num of epochs in one xx min duration (for example, nepoch=30 when 15min)
    '''
    def __init__(self, n_channels, nepoch):
        super(MultiCNNC2CM_S, self).__init__()
        self.multicnn_se = MultiCNN_SE(n_channels)
        self.conv1 = SingleConv(128, 128, kernel_size=(1,7), stride=(1,3), padding=0)
        self.conv2 = SingleConv(128, 128, kernel_size=(1,3), stride=(1,3), padding=0)

        self.conv2_ss1 = SingleConv(128, 5, kernel_size=(1,4), stride=1, padding=0)
        self.out_ss = OutSleepStage(5,5)

        # self.conv3 = SingleConv(128, 256, kernel_size=(nepoch,1), stride=1, padding=0)
        # self.out_d = OutDiagnosis(1024, hidden_channels=256)
    def forward(self, x):
        x = self.multicnn_se(x)
        x = self.conv1(x)
        x = self.conv2(x)

        ss = self.conv2_ss1(x)
        ss = self.out_ss(ss)
        ss = torch.squeeze(ss, 3)

        # d = self.conv3(x)
        # d = self.out_d(d)
        return ss

class MultiCNNC2CM_D(nn.Module):
    '''
    Plan 2: input: 3*30*3000
    - kernel_shape: C2CM (not square)
    - kernel_size: big, small
    - structure: encoder
    combine: SAN (Wei Zhou), C2CM (Cuntai Guan)
    parameter: nepoch: num of epochs in one xx min duration (for example, nepoch=30 when 15min)
    '''
    def __init__(self, n_channels, nepoch):
        super(MultiCNNC2CM_D, self).__init__()
        self.multicnn_se = MultiCNN_SE(n_channels)
        self.conv1 = SingleConv(128, 128, kernel_size=(1,7), stride=(1,3), padding=0)
        self.conv2 = SingleConv(128, 128, kernel_size=(1,3), stride=(1,3), padding=0)

        # self.conv2_ss1 = SingleConv(128, 5, kernel_size=(1,4), stride=1, padding=0)
        # self.out_ss = OutSleepStage(5,5)

        self.conv3 = SingleConv(128, 256, kernel_size=(nepoch,1), stride=1, padding=0)
        self.out_d = OutDiagnosis(1024, hidden_channels=256)
    def forward(self, x):
        x = self.multicnn_se(x)
        x = self.conv1(x)
        x = self.conv2(x)

        # ss = self.conv2_ss1(x)
        # ss = self.out_ss(ss)
        # ss = torch.squeeze(ss, 3)

        d = self.conv3(x)
        d = self.out_d(d)
        return d

class SquareSmall2_5min_D(nn.Module):
    '''(input: 3*100*150)
    Plan 1: 
    - kernel_shape: square
    - kernel_size: small
    - structure: encoder'''
    def __init__(self, n_channels, nepoch): # TODO: change parameters | classifier.py reshape
        super(SquareSmall2_5min_D, self).__init__()
        self.conv1 = SingleConv(n_channels, 64, kernel_size=(3,5), stride=(2,3), padding=1)     # main: conv2d + batchnorm + relu
        self.conv2 = SingleConv(64, 128, kernel_size=5, stride=3, padding=0)            # main: conv2d + batchnorm + relu
        self.conv3 = SingleConv(128, 128, kernel_size=4, stride=3, padding=0)           # main: conv2d + batchnorm + relu

        # self.conv3_ss1 = SingleConv(128, 5, kernel_size=(1,nepoch), stride=1, padding=0)  # sleep stage: conv2d + batchnorm + relu
        # # self.conv3_ss2 = SingleConv(128, 64, kernel_size=1, stride=1, padding=0)        # sleep stage: conv2d + batchnorm + relu
        # self.out_ss = OutSleepStage(5, 5)                                              # sleep stage: conv2d (kernel_size=1)

        self.conv4 = SingleConv(128, 256, kernel_size=3, stride=1, padding=1)           # main: conv2d + batchnorm + relu
        self.conv5 = SingleConv(256, 256, kernel_size=3, stride=1, padding=0)           # main: conv2d + batchnorm + relu
        self.conv6 = SingleConv(256, 256, kernel_size=2, stride=1, padding=0)           # main: conv2d + batchnorm + relu
        
        self.out_d = OutDiagnosis(1024,hidden_channels=256)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # ss = self.conv3_ss1(x)
        # # ss = self.conv3_ss2(ss)
        # ss = self.out_ss(ss)    # multitask 1: sleep staging
        # ss = torch.squeeze(ss, 3) # [TODO] 效果怎么样？

        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        d = self.out_d(x)       # multitask 2: narcolepsy diagnosis
        return d

class SquareSmall2_5min(nn.Module):
    '''(input: 3*100*150)
    Plan 1: 
    - kernel_shape: square
    - kernel_size: small
    - structure: encoder'''
    def __init__(self, n_channels, nepoch): # TODO: change parameters | classifier.py reshape
        super(SquareSmall2_5min, self).__init__()
        self.conv1 = SingleConv(n_channels, 64, kernel_size=(3,5), stride=(2,3), padding=1)     # main: conv2d + batchnorm + relu
        self.conv2 = SingleConv(64, 128, kernel_size=5, stride=3, padding=0)            # main: conv2d + batchnorm + relu
        self.conv3 = SingleConv(128, 128, kernel_size=4, stride=3, padding=0)           # main: conv2d + batchnorm + relu

        self.conv3_ss1 = SingleConv(128, 5, kernel_size=(1,nepoch), stride=1, padding=0)  # sleep stage: conv2d + batchnorm + relu
        # self.conv3_ss2 = SingleConv(128, 64, kernel_size=1, stride=1, padding=0)        # sleep stage: conv2d + batchnorm + relu
        self.out_ss = OutSleepStage(5, 5)                                              # sleep stage: conv2d (kernel_size=1)

        self.conv4 = SingleConv(128, 256, kernel_size=3, stride=1, padding=1)           # main: conv2d + batchnorm + relu
        self.conv5 = SingleConv(256, 256, kernel_size=3, stride=1, padding=0)           # main: conv2d + batchnorm + relu
        self.conv6 = SingleConv(256, 256, kernel_size=2, stride=1, padding=0)           # main: conv2d + batchnorm + relu
        
        self.out_d = OutDiagnosis(1024,hidden_channels=256)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        ss = self.conv3_ss1(x)
        # ss = self.conv3_ss2(ss)
        ss = self.out_ss(ss)    # multitask 1: sleep staging
        ss = torch.squeeze(ss, 3) # [TODO] 效果怎么样？

        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        d = self.out_d(x)       # multitask 2: narcolepsy diagnosis
        return ss, d

class SquareSmall2_5min_S(nn.Module):
    '''(input: 3*100*150)
    Plan 1: 
    - kernel_shape: square
    - kernel_size: small
    - structure: encoder'''
    def __init__(self, n_channels, nepoch): # TODO: change parameters | classifier.py reshape
        super(SquareSmall2_5min_S, self).__init__()
        self.conv1 = SingleConv(n_channels, 64, kernel_size=(3,5), stride=(2,3), padding=1)     # main: conv2d + batchnorm + relu
        self.conv2 = SingleConv(64, 128, kernel_size=5, stride=3, padding=0)            # main: conv2d + batchnorm + relu
        self.conv3 = SingleConv(128, 128, kernel_size=4, stride=3, padding=0)           # main: conv2d + batchnorm + relu

        self.conv3_ss1 = SingleConv(128, 5, kernel_size=(1,nepoch), stride=1, padding=0)  # sleep stage: conv2d + batchnorm + relu
        # self.conv3_ss2 = SingleConv(128, 64, kernel_size=1, stride=1, padding=0)        # sleep stage: conv2d + batchnorm + relu
        self.out_ss = OutSleepStage(5, 5)                                              # sleep stage: conv2d (kernel_size=1)

        # self.conv4 = SingleConv(128, 256, kernel_size=3, stride=1, padding=1)           # main: conv2d + batchnorm + relu
        # self.conv5 = SingleConv(256, 256, kernel_size=3, stride=1, padding=0)           # main: conv2d + batchnorm + relu
        # self.conv6 = SingleConv(256, 256, kernel_size=2, stride=1, padding=0)           # main: conv2d + batchnorm + relu
        
        # self.out_d = OutDiagnosis(1024,hidden_channels=256)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        ss = self.conv3_ss1(x)
        # ss = self.conv3_ss2(ss)
        ss = self.out_ss(ss)    # multitask 1: sleep staging
        ss = torch.squeeze(ss, 3) # [TODO] 效果怎么样？

        # x = self.conv4(x)
        # x = self.conv5(x)
        # x = self.conv6(x)

        # d = self.out_d(x)       # multitask 2: narcolepsy diagnosis
        return ss

class SquareSmall5min(nn.Module):
    '''(input: 3*100*300)
    Plan 1: 
    - kernel_shape: square
    - kernel_size: small
    - structure: encoder'''
    def __init__(self, n_channels, nepoch): # TODO: change parameters | classifier.py reshape
        super(SquareSmall5min, self).__init__()
        self.conv1 = SingleConv(n_channels, 64, kernel_size=(3,5), stride=(1,3), padding=1)     # main: conv2d + batchnorm + relu
        self.conv2 = SingleConv(64, 128, kernel_size=7, stride=3, padding=0)            # main: conv2d + batchnorm + relu
        self.conv3 = SingleConv(128, 128, kernel_size=7, stride=3, padding=1)           # main: conv2d + batchnorm + relu

        self.conv3_ss1 = SingleConv(128, 5, kernel_size=(1,nepoch), stride=1, padding=0)  # sleep stage: conv2d + batchnorm + relu
        # self.conv3_ss2 = SingleConv(128, 64, kernel_size=1, stride=1, padding=0)        # sleep stage: conv2d + batchnorm + relu
        self.out_ss = OutSleepStage(5, 5)                                              # sleep stage: conv2d (kernel_size=1)

        self.conv4 = SingleConv(128, 256, kernel_size=3, stride=1, padding=1)           # main: conv2d + batchnorm + relu
        self.conv5 = SingleConv(256, 256, kernel_size=3, stride=1, padding=0)           # main: conv2d + batchnorm + relu
        self.conv6 = SingleConv(256, 256, kernel_size=3, stride=1, padding=0)           # main: conv2d + batchnorm + relu
        self.conv7 = SingleConv(256, 256, kernel_size=3, stride=1, padding=0)           # main: conv2d + batchnorm + relu
        self.conv8 = SingleConv(256, 256, kernel_size=3, stride=1, padding=0)           # main: conv2d + batchnorm + relu
        
        self.out_d = OutDiagnosis(1024,hidden_channels=256)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        ss = self.conv3_ss1(x)
        # ss = self.conv3_ss2(ss)
        ss = self.out_ss(ss)    # multitask 1: sleep staging
        ss = torch.squeeze(ss, 3) # [TODO] 效果怎么样？

        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)

        d = self.out_d(x)       # multitask 2: narcolepsy diagnosis
        return ss, d

class SquareSmall10min(nn.Module):
    '''(input: 3*200*300)
    Plan 1: 
    - kernel_shape: square
    - kernel_size: small
    - structure: encoder'''
    def __init__(self, n_channels, nepoch): # TODO: change parameters | classifier.py reshape
        super(SquareSmall10min, self).__init__()
        self.conv1 = SingleConv(n_channels, 64, kernel_size=(3,5), stride=(2,3), padding=1)     # main: conv2d + batchnorm + relu
        self.conv2 = SingleConv(64, 128, kernel_size=5, stride=2, padding=1)            # main: conv2d + batchnorm + relu
        self.conv3 = SingleConv(128, 128, kernel_size=7, stride=2, padding=0)
        self.conv4 = SingleConv(128, 128, kernel_size=3, stride=1, padding=0)           # main: conv2d + batchnorm + relu

        self.conv4_ss1 = SingleConv(128, 5, kernel_size=(1,nepoch), stride=1, padding=0)  # sleep stage: conv2d + batchnorm + relu
        # self.conv3_ss2 = SingleConv(128, 64, kernel_size=1, stride=1, padding=0)        # sleep stage: conv2d + batchnorm + relu
        self.out_ss = OutSleepStage(5, 5)                                              # sleep stage: conv2d (kernel_size=1)

        self.conv5 = SingleConv(128, 256, kernel_size=3, stride=2, padding=1)           # main: conv2d + batchnorm + relu
        self.conv6 = SingleConv(256, 256, kernel_size=3, stride=1, padding=0)           # main: conv2d + batchnorm + relu
        self.conv7 = SingleConv(256, 256, kernel_size=3, stride=1, padding=0)           # main: conv2d + batchnorm + relu
        self.conv8 = SingleConv(256, 256, kernel_size=3, stride=1, padding=0)           # main: conv2d + batchnorm + relu
        self.conv9 = SingleConv(256, 256, kernel_size=3, stride=1, padding=0)           # main: conv2d + batchnorm + relu
        
        self.out_d = OutDiagnosis(1024,hidden_channels=256)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        ss = self.conv4_ss1(x)
        # ss = self.conv3_ss2(ss)
        ss = self.out_ss(ss)    # multitask 1: sleep staging
        ss = torch.squeeze(ss, 3) # [TODO] 效果怎么样？

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)

        d = self.out_d(x)       # multitask 2: narcolepsy diagnosis
        return ss, d

class SquareSmall15min(nn.Module):
    '''(input: 3*300*300)
    Plan 1: 
    - kernel_shape: square
    - kernel_size: small
    - structure: encoder'''
    def __init__(self, n_channels, nepoch):
        super(SquareSmall15min, self).__init__()
        self.conv1 = SingleConv(n_channels, 64, kernel_size=5, stride=3, padding=1)     # main: conv2d + batchnorm + relu
        self.conv2 = SingleConv(64, 128, kernel_size=7, stride=3, padding=0)            # main: conv2d + batchnorm + relu
        self.conv3 = SingleConv(128, 128, kernel_size=3, stride=1, padding=0)           # main: conv2d + batchnorm + relu

        self.conv3_ss1 = SingleConv(128, 5, kernel_size=(1,nepoch), stride=1, padding=0)  # sleep stage: conv2d + batchnorm + relu
        # self.conv3_ss2 = SingleConv(128, 64, kernel_size=1, stride=1, padding=0)        # sleep stage: conv2d + batchnorm + relu
        self.out_ss = OutSleepStage(5, 5)                                              # sleep stage: conv2d (kernel_size=1)

        self.conv4 = SingleConv(128, 256, kernel_size=5, stride=3, padding=1)           # main: conv2d + batchnorm + relu
        self.conv5 = SingleConv(256, 256, kernel_size=3, stride=1, padding=0)           # main: conv2d + batchnorm + relu
        self.conv6 = SingleConv(256, 256, kernel_size=3, stride=1, padding=0)           # main: conv2d + batchnorm + relu
        self.conv7 = SingleConv(256, 256, kernel_size=3, stride=1, padding=0)           # main: conv2d + batchnorm + relu
        self.conv8 = SingleConv(256, 256, kernel_size=3, stride=1, padding=0)           # main: conv2d + batchnorm + relu
        
        self.out_d = OutDiagnosis(1024,hidden_channels=256)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        ss = self.conv3_ss1(x)
        # ss = self.conv3_ss2(ss)
        ss = self.out_ss(ss)    # multitask 1: sleep staging
        ss = torch.squeeze(ss, 3) # [TODO] 效果怎么样？

        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)

        d = self.out_d(x)       # multitask 2: narcolepsy diagnosis
        return ss, d

class SquareSmall30min(nn.Module):
    '''(input: 3*600*300)
    Plan 1: 
    - kernel_shape: square
    - kernel_size: small
    - structure: encoder'''
    def __init__(self, n_channels, nepoch): # TODO: change parameters | classifier.py reshape
        super(SquareSmall30min, self).__init__()
        self.conv1 = SingleConv(n_channels, 64, kernel_size=(5,7), stride=(3,2), padding=(1,0))     # main: conv2d + batchnorm + relu
        self.conv2 = SingleConv(64, 128, kernel_size=(5,7), stride=(3,2), padding=(0,1))            # main: conv2d + batchnorm + relu
        self.conv3 = SingleConv(128, 128, kernel_size=(5,7), stride=1, padding=0)
        self.conv4 = SingleConv(128, 128, kernel_size=(3,7), stride=1, padding=0)           # main: conv2d + batchnorm + relu

        self.conv4_ss1 = SingleConv(128, 5, kernel_size=(1,nepoch), stride=1, padding=0)  # sleep stage: conv2d + batchnorm + relu
        # self.conv3_ss2 = SingleConv(128, 64, kernel_size=1, stride=1, padding=0)        # sleep stage: conv2d + batchnorm + relu
        self.out_ss = OutSleepStage(5, 5)                                              # sleep stage: conv2d (kernel_size=1)

        self.conv5 = SingleConv(128, 256, kernel_size=3, stride=3, padding=0)           # main: conv2d + batchnorm + relu
        self.conv6 = SingleConv(256, 256, kernel_size=5, stride=2, padding=0)           # main: conv2d + batchnorm + relu
        self.conv7 = SingleConv(256, 256, kernel_size=3, stride=1, padding=0)           # main: conv2d + batchnorm + relu
        self.conv8 = SingleConv(256, 256, kernel_size=3, stride=1, padding=0)           # main: conv2d + batchnorm + relu
        self.conv9 = SingleConv(256, 256, kernel_size=3, stride=1, padding=0)           # main: conv2d + batchnorm + relu
        
        self.out_d = OutDiagnosis(1024,hidden_channels=256)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        ss = self.conv4_ss1(x)
        # ss = self.conv3_ss2(ss)
        ss = self.out_ss(ss)    # multitask 1: sleep staging
        ss = torch.squeeze(ss, 3) # [TODO] 效果怎么样？

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)

        d = self.out_d(x)       # multitask 2: narcolepsy diagnosis
        return ss, d
        
class SquareSmall60min(nn.Module):
    '''(input: 3*600*600)
    Plan 1: 
    - kernel_shape: square
    - kernel_size: small
    - structure: encoder'''
    def __init__(self, n_channels, nepoch):
        super(SquareSmall60min, self).__init__()
        self.conv1 = SingleConv(n_channels, 64, kernel_size=7, stride=5, padding=1)     # main: conv2d + batchnorm + relu
        self.conv2 = SingleConv(64, 128, kernel_size=3, stride=1, padding=1)            # main: conv2d + batchnorm + relu
        self.conv3 = SingleConv(128, 128, kernel_size=3, stride=1, padding=1)           # main: conv2d + batchnorm + relu

        self.conv3_ss1 = SingleConv(128, 5, kernel_size=(1,nepoch), stride=1, padding=0)  # sleep stage: conv2d + batchnorm + relu
        # self.conv3_ss2 = SingleConv(128, 64, kernel_size=1, stride=1, padding=0)        # sleep stage: conv2d + batchnorm + relu
        self.out_ss = OutSleepStage(5, 5)                                              # sleep stage: conv2d (kernel_size=1)

        self.conv4 = SingleConv(128, 256, kernel_size=3, stride=3, padding=0)           # main: conv2d + batchnorm + relu
        self.conv5 = SingleConv(256, 256, kernel_size=3, stride=3, padding=1)           # main: conv2d + batchnorm + relu
        self.conv6 = SingleConv(256, 256, kernel_size=3, stride=2, padding=0)           # main: conv2d + batchnorm + relu
        self.conv7 = SingleConv(256, 256, kernel_size=3, stride=1, padding=0)           # main: conv2d + batchnorm + relu
        self.conv8 = SingleConv(256, 256, kernel_size=3, stride=1, padding=0)           # main: conv2d + batchnorm + relu
        
        self.out_d = OutDiagnosis(1024,hidden_channels=256)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        ss = self.conv3_ss1(x)
        # ss = self.conv3_ss2(ss)
        ss = self.out_ss(ss)    # multitask 1: sleep staging
        ss = torch.squeeze(ss, 3) # [TODO] 效果怎么样？

        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)

        d = self.out_d(x)       # multitask 2: narcolepsy diagnosis
        return ss, d

class SquareSmall90min(nn.Module):
    '''(input: 3*900*600)
    Plan 1: 
    - kernel_shape: square
    - kernel_size: small
    - structure: encoder'''
    def __init__(self, n_channels, nepoch):
        super(SquareSmall90min, self).__init__()
        self.conv1 = SingleConv(n_channels, 64, kernel_size=(7,5), stride=(5,3), padding=1)     # main: conv2d + batchnorm + relu
        self.conv2 = SingleConv(64, 128, kernel_size=(3,7), stride=1, padding=(1,0))            # main: conv2d + batchnorm + relu
        self.conv3 = SingleConv(128, 128, kernel_size=(3,7), stride=1, padding=(1,0))
        self.conv4 = SingleConv(128, 128, kernel_size=(3,7), stride=1, padding=(1,0))
        self.conv5 = SingleConv(128, 128, kernel_size=3, stride=1, padding=(1,0))           # main: conv2d + batchnorm + relu

        self.conv5_ss1 = SingleConv(128, 5, kernel_size=(1,nepoch), stride=1, padding=0)  # sleep stage: conv2d + batchnorm + relu
        # self.conv3_ss2 = SingleConv(128, 64, kernel_size=1, stride=1, padding=0)        # sleep stage: conv2d + batchnorm + relu
        self.out_ss = OutSleepStage(5, 5)                                              # sleep stage: conv2d (kernel_size=1)

        self.conv6 = SingleConv(128, 256, kernel_size=7, stride=5, padding=1)           # main: conv2d + batchnorm + relu
        self.conv7 = SingleConv(256, 256, kernel_size=3, stride=3, padding=0)           # main: conv2d + batchnorm + relu
        self.conv8 = SingleConv(256, 256, kernel_size=3, stride=3, padding=0)           # main: conv2d + batchnorm + relu
        self.conv9 = SingleConv(256, 256, kernel_size=3, stride=1, padding=0)           # main: conv2d + batchnorm + relu
        
        self.out_d = OutDiagnosis(1024,hidden_channels=256)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        ss = self.conv5_ss1(x)
        # ss = self.conv3_ss2(ss)
        ss = self.out_ss(ss)    # multitask 1: sleep staging
        ss = torch.squeeze(ss, 3) # [TODO] 效果怎么样？

        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)

        d = self.out_d(x)       # multitask 2: narcolepsy diagnosis
        return ss, d

class SquareSmall90min_D(nn.Module):
    '''(input: 3*900*600)
    Plan 1: 
    - kernel_shape: square
    - kernel_size: small
    - structure: encoder'''
    def __init__(self, n_channels, nepoch):
        super(SquareSmall90min_D, self).__init__()
        self.conv1 = SingleConv(n_channels, 64, kernel_size=(7,5), stride=(5,3), padding=1)     # main: conv2d + batchnorm + relu
        self.conv2 = SingleConv(64, 128, kernel_size=(3,7), stride=1, padding=(1,0))            # main: conv2d + batchnorm + relu
        self.conv3 = SingleConv(128, 128, kernel_size=(3,7), stride=1, padding=(1,0))
        self.conv4 = SingleConv(128, 128, kernel_size=(3,7), stride=1, padding=(1,0))
        self.conv5 = SingleConv(128, 128, kernel_size=3, stride=1, padding=(1,0))           # main: conv2d + batchnorm + relu

        # self.conv5_ss1 = SingleConv(128, 5, kernel_size=(1,nepoch), stride=1, padding=0)  # sleep stage: conv2d + batchnorm + relu
        # # self.conv3_ss2 = SingleConv(128, 64, kernel_size=1, stride=1, padding=0)        # sleep stage: conv2d + batchnorm + relu
        # self.out_ss = OutSleepStage(5, 5)                                              # sleep stage: conv2d (kernel_size=1)

        self.conv6 = SingleConv(128, 256, kernel_size=7, stride=5, padding=1)           # main: conv2d + batchnorm + relu
        self.conv7 = SingleConv(256, 256, kernel_size=3, stride=3, padding=0)           # main: conv2d + batchnorm + relu
        self.conv8 = SingleConv(256, 256, kernel_size=3, stride=3, padding=0)           # main: conv2d + batchnorm + relu
        self.conv9 = SingleConv(256, 256, kernel_size=3, stride=1, padding=0)           # main: conv2d + batchnorm + relu
        
        self.out_d = OutDiagnosis(1024,hidden_channels=256)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        # ss = self.conv5_ss1(x)
        # # ss = self.conv3_ss2(ss)
        # ss = self.out_ss(ss)    # multitask 1: sleep staging
        # ss = torch.squeeze(ss, 3) # [TODO] 效果怎么样？

        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)

        d = self.out_d(x)       # multitask 2: narcolepsy diagnosis
        return d

class SquareSmall90min_S(nn.Module):
    '''(input: 3*900*600)
    Plan 1: 
    - kernel_shape: square
    - kernel_size: small
    - structure: encoder'''
    def __init__(self, n_channels, nepoch):
        super(SquareSmall90min_S, self).__init__()
        self.conv1 = SingleConv(n_channels, 64, kernel_size=(7,5), stride=(5,3), padding=1)     # main: conv2d + batchnorm + relu
        self.conv2 = SingleConv(64, 128, kernel_size=(3,7), stride=1, padding=(1,0))            # main: conv2d + batchnorm + relu
        self.conv3 = SingleConv(128, 128, kernel_size=(3,7), stride=1, padding=(1,0))
        self.conv4 = SingleConv(128, 128, kernel_size=(3,7), stride=1, padding=(1,0))
        self.conv5 = SingleConv(128, 128, kernel_size=3, stride=1, padding=(1,0))           # main: conv2d + batchnorm + relu

        self.conv5_ss1 = SingleConv(128, 5, kernel_size=(1,nepoch), stride=1, padding=0)  # sleep stage: conv2d + batchnorm + relu
        # self.conv3_ss2 = SingleConv(128, 64, kernel_size=1, stride=1, padding=0)        # sleep stage: conv2d + batchnorm + relu
        self.out_ss = OutSleepStage(5, 5)                                              # sleep stage: conv2d (kernel_size=1)

        # self.conv6 = SingleConv(128, 256, kernel_size=7, stride=5, padding=1)           # main: conv2d + batchnorm + relu
        # self.conv7 = SingleConv(256, 256, kernel_size=3, stride=3, padding=0)           # main: conv2d + batchnorm + relu
        # self.conv8 = SingleConv(256, 256, kernel_size=3, stride=3, padding=0)           # main: conv2d + batchnorm + relu
        # self.conv9 = SingleConv(256, 256, kernel_size=3, stride=1, padding=0)           # main: conv2d + batchnorm + relu
        
        # self.out_d = OutDiagnosis(1024,hidden_channels=256)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        ss = self.conv5_ss1(x)
        # ss = self.conv3_ss2(ss)
        ss = self.out_ss(ss)    # multitask 1: sleep staging
        ss = torch.squeeze(ss, 3) # [TODO] 效果怎么样？

        # x = self.conv6(x)
        # x = self.conv7(x)
        # x = self.conv8(x)
        # x = self.conv9(x)

        # d = self.out_d(x)       # multitask 2: narcolepsy diagnosis
        return ss