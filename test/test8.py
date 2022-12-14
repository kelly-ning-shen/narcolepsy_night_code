import torch
import torch.nn as nn
import torch.nn.functional as F

class SpConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,*args,**kwargs):
        super(SpConv2d,self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding)
    def forward(self,x):
        n,c,h,w = x.size()
        assert c % 4 == 0 # need 4*n channels (can't)
        x1 = x[:,:c//4,:,:]
        x2 = x[:,c//4:c//2,:,:]
        x3 = x[:,c//2:c//4*3,:,:]
        x4 = x[:,c//4*3:c,:,:]
        x1 = nn.functional.pad(x1,(1,0,1,0),mode = "constant",value = 0) # left top
        x2 = nn.functional.pad(x2,(0,1,1,0),mode = "constant",value = 0) # right top
        x3 = nn.functional.pad(x3,(1,0,0,1),mode = "constant",value = 0) # left bottom
        x4 = nn.functional.pad(x4,(0,1,0,1),mode = "constant",value = 0) # right bottom
        x = torch.cat([x1,x2,x3,x4],dim = 1)
        return self.conv(x) # [10,4,301,301]
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
    '''squeeze-and-excitation networks, Momenta, ImageNet2017 champiom'''
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
        x = torch.cat((x1, x2), dim=3) # TODO: concat ????????????????????????
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
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        # self.conv = SpConv2d(3,16,4,1,0)
        # self.pool = nn.MaxPool2d(kernel_size=(1,4),stride=(1,4))
        # self.conv = SpConv2d(in_channels=4,out_channels=16,kernel_size=4,stride=1,padding=0)
        self.se = SELayer(128, 16)
    def forward(self,x):
        return self.se(x)

class MultiCNNC2CM(nn.Module):
    '''
    Plan 2: input: 3*30*3000
    - kernel_shape: C2CM (not square)
    - kernel_size: big, small
    - structure: encoder
    combine: SAN (Wei Zhou), C2CM (Cuntai Guan)
    '''
    def __init__(self, n_channels):
        super(MultiCNNC2CM, self).__init__()
        self.multicnn_se = MultiCNN_SE(n_channels)
        self.conv1 = SingleConv(128, 128, kernel_size=(1,7), stride=(1,3), padding=0)
        self.conv2 = SingleConv(128, 128, kernel_size=(1,3), stride=(1,3), padding=0)

        self.conv2_ss1 = SingleConv(128, 5, kernel_size=(1,4), stride=1, padding=0)
        self.out_ss = OutSleepStage(5,5)

        self.conv3 = SingleConv(128, 256, kernel_size=(30,1), stride=1, padding=0)
        self.out_d = OutDiagnosis(1024, hidden_channels=256)
    def forward(self, x):
        x = self.multicnn_se(x)
        x = self.conv1(x)
        x = self.conv2(x)

        ss = self.conv2_ss1(x)
        ss = self.out_ss(ss)
        ss = torch.squeeze(ss, 3)

        d = self.conv3(x)
        # d = self.out_d(d)
        return ss, d

if __name__ == "__main__":
    x = torch.randn(10,3,30,3000)
    # net = Net()
    # net = MultiCNN_SE(in_channels=3)
    net = MultiCNNC2CM(n_channels=3)
    ss, d = net(x)
    print(ss)
    print(ss.shape)
    print(d)
    print(d.shape)