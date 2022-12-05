'''
Leave-one-subject-out (LOSO)
input: 15 min (EEG, EOG, EMG) shape: 3*300*300
output: (multitask) for 15 min: 1. sleep stages; 2. diagnosis ["Other","Narcolepsy type 1"]
'''
import numpy as np
import pickle
from pathlib import Path
from tkinter import _flatten
from operator import itemgetter
import matplotlib.pyplot as plt
import os
import sys
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# from tqdm import tqdm

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Function
from torch.utils.data import Dataset, DataLoader

from a_tools import myprint
from confusion_matrix_index import plot_confusion_matrix

savelog = 1
savepic = 1
savecheckpoints = 1
MODE = 'squaresmalle_15min_zscore_shuffle'

if savelog:
    class Logger(object):
        def __init__(self, filename="Default.log"):
            self.terminal = sys.stdout
            self.log = open(filename, "a")
    
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass
    
    path = os.path.abspath(os.path.dirname(__file__))
    type = sys.getfilesystemencoding()
    # sys.stdout = Logger('log/withoutIH_AASM_right_IIRFil0.3_' + feature_type + '_nol.txt') # 不需要自己先新建txt文档  # right: filter_right
    sys.stdout = Logger(f'log/TEST_classifier_{MODE}.txt') # 不需要自己先新建txt文档  # right: filter_right

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

base = 'data/mnc/cnc/cnc/'
BATCH_SIZE = 10
EPOCHS = 10 # i.e, 10
LR = 0.001        # i.e, 0.001

# prepare checkpoint dir
dir_checkpoint = f'checkpoints/{MODE}/checkpoints'
DIAGNOSIS = ["Other","Narcolepsy type 1"]

def loadSubjectData(base):
    '''
    return filepaths of train and test
    '''
    subjects = sorted(Path(base).glob(f'*.xml'))
    nsubject = len(subjects)
    print(f'Subjects num: {nsubject}\n')

    subjects_data = {}
    # read: the inputs of one subject (save to dict)
    for i in range(nsubject):
        subject = subjects[i] # WindowsPath('dsata/mnc/cnc/cnc/chc001-nsrr.xml')
        name = subject.stem
        subject_data = sorted(Path(base).glob(f'{name}_15min_zscore_*.s_pkl'))
        subject_data = sorted(subject_data, key=lambda x: int(str(x).split('.')[0].split('_')[-1]))
        ndata = len(subject_data)
        myprint(f'15min inputs: {name} ({ndata})')
        subjects_data[name] = subject_data # list of datapath
    
    return subjects_data

def LeaveOneSubjectOut(base):
    subjects_data = loadSubjectData(base)
    conf_mats = np.zeros((5,5), dtype=np.int)
    for subject in subjects_data:
        # get filepaths of train_data, test_data
        # Can use these filepaths for CustomDataset
        test_data = subjects_data[subject] # get filepaths of test data

        train_subjects = list(subjects_data.keys()) # get filepaths of train data
        train_subjects.remove(subject)
        train_data = list(_flatten(itemgetter(*train_subjects)(subjects_data)))

        ntrain = len(train_data)
        ntest = len(test_data)

        print(f'\n=== Test on {subject}. train_data({ntrain}), test_data({ntest}) ===')
        print('Define dataloader')

        # train_dataset = NarcoNight15min(train_data)
        # test_dataset = NarcoNight15min(test_data)

        print('==== START TRAINING ====')
        model = SquareSmallE(n_channels=3)
        # if torch.cuda.device_count()>1:
        #     model = nn.DataParallel(model)
        model.to(device)
        diagnose_loss = nn.BCEWithLogitsLoss()
        sleepstage_loss = nn.CrossEntropyLoss(ignore_index=-1) # ignore sleep stage ann with -1
        optimizer = optim.Adam(model.parameters(), lr=LR)

        for epoch in range(EPOCHS):
            print('Starting epoch {}/{}.'.format(epoch + 1, EPOCHS)) 
            model.train()
            train_dataloader = DataLoader(NarcoNight15min(train_data), shuffle=True, batch_size=BATCH_SIZE)
            epoch_loss = 0

            for i, data in enumerate(train_dataloader): # tqdm(train_dataloader)
                inputs = data['signal_pic'].to(device) # shape: [10,3,300,300]
                
                ss_labels = data['ann'].to(device) # shape: [10,30]
                d_labels = data['diagnosis'].to(device) # shape: [10]

                ss_outputs, d_outputs = model(inputs) # ss_outputs: shape: [10,5,30]; d_outputs: shape: [10,1]

                loss_ss = sleepstage_loss(ss_outputs, ss_labels) # preds=ss_outputs [10,5,30]; labels=ss_labels [10,30]
                loss_d = diagnose_loss(d_outputs, d_labels.unsqueeze(1).float()) # shape: [10,1]

                loss = loss_ss + loss_d #  [TODO] multitask 中简单相加loss，肯定是不合理的，还需要修改！
                epoch_loss += loss.item()
                if i % 10 == 0: # print loss every 10 step
                    print('{0:.4f} --- loss: {1:.6f}, loss_ss: {2:.6f}, loss_d: {3:.6f}'.format(i * BATCH_SIZE / ntrain, loss.item(), loss_ss.item(), loss_d.item()))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


            print('Epoch finished! Loss: {}'.format(epoch_loss/i))
            if savecheckpoints:
                torch.save(model.state_dict(),
                            f'{dir_checkpoint}_{subject}_CP{epoch+1}.pth')
                            # dir_checkpoint + 'CP{}.pth'.format(epoch + 1))
            print('Checkpoint {} saved !'.format(epoch + 1))
        
        print('==== START TESTING ====')
        # [TODO] test_loader是否需要使用minibatch？
        test_dataloader = DataLoader(NarcoNight15min(test_data), shuffle=False, batch_size=BATCH_SIZE)
        conf_mat = test_on_subject(model, test_dataloader, ntest, subject)
        conf_mats += conf_mat
    picpath = f'pic/{MODE}/conf_mat_all.png' # TODO: 根据运行设置将图片放到某个文件夹里
    plot_confusion_matrix(conf_mats,5,'all',savepic=savepic,picpath=picpath)

class NarcoNight15min(Dataset):
    def __init__(self, filepaths):
        self.filepaths = filepaths
        self.data = []
        self.diagnosis = []
        for filepath in filepaths:
            self.data.append(filepath)
            diagnosis = filepath.stem[2]
            if diagnosis == 'c': # for CNC only
                self.diagnosis.append(0)
            else:
                self.diagnosis.append(1)
    
    def __len__(self):
        return len(self.diagnosis)

    def __getitem__(self, index: int):
        d = self.data[index]
        with d.open('rb') as fp:
            signal_pic, ann = pickle.load(fp)
        signal_pic = torch.from_numpy(signal_pic) # shape: torch.Size ([3,300,300])
        ann = torch.from_numpy(ann) # shape: torch.Size ([30])
        diagnosis = self.diagnosis[index]
        sample = {'signal_pic': signal_pic, 'ann': ann, 'diagnosis': diagnosis}
        return sample

####################### DEFINE MODEL #######################
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

class SquareSmallE(nn.Module):
    '''Plan 1: 
    - kernel_shape: square
    - kernel_size: small
    - structure: encoder'''
    def __init__(self, n_channels):
        super(SquareSmallE, self).__init__()
        self.conv1 = SingleConv(n_channels, 64, kernel_size=5, stride=3, padding=1)     # main: conv2d + batchnorm + relu
        self.conv2 = SingleConv(64, 128, kernel_size=7, stride=3, padding=0)            # main: conv2d + batchnorm + relu
        self.conv3 = SingleConv(128, 128, kernel_size=3, stride=1, padding=0)           # main: conv2d + batchnorm + relu

        self.conv3_ss1 = SingleConv(128, 5, kernel_size=(1,30), stride=1, padding=0)  # sleep stage: conv2d + batchnorm + relu
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

def test_on_subject(model, dataloader, ntest, subject):
    # set net model to evaluation
    model.eval() # 不启用 BatchNormalization 和 Dropout
    sss = np.zeros((ntest*30,2)) # metric 1 (sleep stage). col0: preds, col1: lables (15min: 30 epochs)
    ds = np.zeros((ntest,2)) # metric 2 (diagnose). col0: preds, col1: lables
    for i, data in enumerate(dataloader): # tqdm()
        # Get data from the batch
        inputs = data['signal_pic'].to(device) # shape: [10,3,300,300]
                
        ss_labels = data['ann'].to(device) # shape: [10,30]
        d_labels = data['diagnosis'].to(device) # shape: [10]

        ss_outputs, d_outputs = model(inputs) # ss_outputs: shape: [10,5,30]; d_outputs: shape: [10,1]
        # metric 1: sleep stage (predicted and true anns for the whole night)
        ## import confusion_matrix_index (get cm, acc, sen, spec)
        # softmax = nn.Softmax(dim=1)
        ss_outputs = F.softmax(ss_outputs, dim=1)
        ss_outputs_indices = torch.argmax(ss_outputs, dim=1) # shape: [10,30]
        nbatch = d_labels.shape[0]
        sss[i*BATCH_SIZE*30:i*BATCH_SIZE*30+nbatch*30, 0] = myflatten(ss_outputs_indices.cpu().numpy()) # col0: preds
        sss[i*BATCH_SIZE*30:i*BATCH_SIZE*30+nbatch*30, 1] = myflatten(data['ann'].numpy()) # col1: lables

        # metric 2: narcolepsy detection (predicted condition for every 15min (for every input))
        ## 编写函数将15min的患病情况转换为整夜的患病情况
        ## return 模型对该患者的判断 （0: control, 1: NT1)
        # sig = nn.Sigmoid()
        d_outputs = F.sigmoid(d_outputs)
        d_outputs = torch.squeeze(d_outputs)
        ds[i*BATCH_SIZE:i*BATCH_SIZE+nbatch, 0] = d_outputs.cpu().detach().numpy() # col0: preds
        ds[i*BATCH_SIZE:i*BATCH_SIZE+nbatch, 1] = data['diagnosis'].numpy() # col1: lables

    print('hello')
    sss = ignore_unknown_label(sss) # ignore sleep stage annotation -1
    acc_ss = accuracy_score(sss[:,1], sss[:,0]) # true, pred
    conf_mat = confusion_matrix(sss[:,1], sss[:,0]) # TODO: 可不可以忽略-1？
    print(f'Sleep stage: acc = {acc_ss}')
    print(classification_report(sss[:,1], sss[:,0]),'\n')
    picpath = f'pic/{MODE}/conf_mat_{subject}.png'
    plot_confusion_matrix(conf_mat,5,subject,savepic=savepic,picpath=picpath)

    acc_d = accuracy_score(ds[:,1], np.where(ds[:,0]>0.5, 1, 0))
    d_pred, d_label = get_diagnose(ds)
    if abs(d_label-d_pred) < 0.5:
        print(f'Right! Diagnosis: {DIAGNOSIS[d_label]}')
    else:
        print(f'Wrong!!! Real Diagnosis: {DIAGNOSIS[d_label]}')
    return conf_mat

def get_diagnose(ds):
    label = int(ds[0,1])

    preds = ds[:,0]
    pred = preds[0]

    if not np.all(preds==pred):
        print(preds)
        pred = np.mean(preds)
    print(f'pred: {pred}, label: {label}')
    return pred, label
    
def myflatten(x):
    x = x.reshape(-1,1)
    x = np.squeeze(x)
    return x

def ignore_unknown_label(sss):
    # ignore epoch with unknown label (-1)
    labels = sss[:,1]
    # preds = sss[:,0]
    idx = np.where(labels==-1)
    sss = np.delete(sss, idx, axis=0)
    return sss


if __name__ == '__main__':    
    LeaveOneSubjectOut(base)
    # x = torch.randn(10,3,300,300)
    # net = SquareSmallE(n_channels=3)
    # ss, d = net(x)
    # print(ss)
    # print(ss.shape) # torch.Size([10, 1, 30, 1])

    # print(d)
    # print(d.shape) # torch.Size([10, 1])