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
# from torch.autograd import Function
from torch.utils.data import Dataset, DataLoader

# from network import SquareSmallE
from network import MultiCNNC2CM
from a_tools import myprint
from a_metrics import plot_confusion_matrix, plot_ROC_curve

savelog = 1
savepic = 1
savecheckpoints = 1
MODE = 'multicnnc2cm_15min_zscore_shuffle_ROC'
is_per_epoch = 1

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
DIAGNOSIS = ["Other","NT1"]
SLEEPSTAGE = ['Wake','N1','N2','N3','REM']

def loadSubjectData(base):
    '''
    return filepaths of train and test
    '''
    subjects = sorted(Path(base).glob(f'*.xml'))
    nsubject = len(subjects)
    print(f'Subjects num: {nsubject}\n')

    subjects_data = {}
    n15minduration = 0
    # read: the inputs of one subject (save to dict)
    for i in range(nsubject):
        subject = subjects[i] # WindowsPath('dsata/mnc/cnc/cnc/chc001-nsrr.xml')
        name = subject.stem
        subject_data = sorted(Path(base).glob(f'{name}_15min_zscore_*.s_pkl'))
        subject_data = sorted(subject_data, key=lambda x: int(str(x).split('.')[0].split('_')[-1]))
        ndata = len(subject_data)
        myprint(f'15min inputs: {name} ({ndata})')
        subjects_data[name] = subject_data # list of datapath
        n15minduration += len(subject_data)
    
    return subjects_data, n15minduration

def LeaveOneSubjectOut(base):
    subjects_data, n15minduration = loadSubjectData(base)
    conf_mats = np.zeros((5,5), dtype=np.int)
    ds_subject = np.zeros((len(subjects_data),3)) # col0: preds (float), col1: lables (int 0,1), col2: preds (int 0,1)
    ds_15min = np.zeros((n15minduration,2)) # [every 15 min] metric 2 (diagnose). col0: pred_probas, col1: lables
    j = -1
    tmp = 0
    for subject in subjects_data:
        j += 1
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
        model = MultiCNNC2CM(n_channels=3)
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
        conf_mat, d_pred, d_label, ds_15min_subject = test_on_subject(model, test_dataloader, ntest, subject)
        ds_15min[tmp:tmp+ntest,:] = ds_15min_subject
        conf_mats += conf_mat
        tmp += ntest
        
        ds_subject[j,0] = d_pred
        ds_subject[j,1] = d_label

        with open(f'diagnosis/{MODE}/ds_15min_subject.txt','a') as fp:
            np.savetxt(fp,np.squeeze(ds_15min_subject[:,0]), fmt='%f', newline=' ')
            fp.write('\n')
            print('Save 15mins of subject {subject}')
    picpath = f'pic/{MODE}/conf_mat_all.png' # TODO: 根据运行设置将图片放到某个文件夹里
    plot_confusion_matrix(conf_mats,5,'all',ticks=SLEEPSTAGE,savepic=savepic,picpath=picpath)
    
    ds_subject[:,2] = np.where(ds_subject[:,0]>0.5, 1, 0)
    acc_ds = accuracy_score(ds_subject[:,1], ds_subject[:,2])
    print(f'Diagnosis acc on patients: {acc_ds}')
    conf_mat_d = confusion_matrix(ds_subject[:,1], ds_subject[:,2]) # true, pred
    picpath = f'pic/{MODE}/conf_mat_diagnosis.png' # TODO: 根据运行设置将图片放到某个文件夹里
    plot_confusion_matrix(conf_mat_d,2,'all',ticks=DIAGNOSIS,savepic=savepic,picpath=picpath) # TODO: 画图吗？还是添加变量改xlabel，ylabel？
    
    # ROC curve
    picpath = f'pic/{MODE}/ROC_curve_diagnosis_subjects.png'
    plot_ROC_curve(ds_subject[:,1],ds_subject[:,0],'all subjects',savepic=savepic,picpath=picpath)
    picpath = f'pic/{MODE}/ROC_curve_diagnosis_15min.png'
    plot_ROC_curve(ds_15min[:,1],ds_15min[:,0],'all 15min',savepic=savepic,picpath=picpath)

    np.savetxt(f'diagnosis/{MODE}/ds_subject.txt', ds_subject, fmt=['%f', '%d', '%d']) # col0: preds (float), col1: lables (int 0,1), col2: preds (int 0,1)
    np.savetxt(f'diagnosis/{MODE}/ds_15min.txt', ds_15min, fmt=['%f', '%d']) # [every 15 min] metric 2 (diagnose). col0: pred_probas, col1: lables

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
        if is_per_epoch:
            signal_pic = signal_pic.reshape((3,30,3000))
        signal_pic = torch.from_numpy(signal_pic) # shape: torch.Size ([3,300,300])
        ann = torch.from_numpy(ann) # shape: torch.Size ([30])
        diagnosis = self.diagnosis[index]
        sample = {'signal_pic': signal_pic, 'ann': ann, 'diagnosis': diagnosis}
        return sample



def test_on_subject(model, dataloader, ntest, subject):
    # set net model to evaluation
    model.eval() # 不启用 BatchNormalization 和 Dropout
    sss = np.zeros((ntest*30,2)) # [every 30s-epoch] metric 1 (sleep stage). col0: preds, col1: lables (15min: 30 epochs)
    ds = np.zeros((ntest,2)) # [every 15 min] metric 2 (diagnose). col0: pred_probas, col1: lables
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
    plot_confusion_matrix(conf_mat,5,subject,ticks=SLEEPSTAGE,savepic=savepic,picpath=picpath)

    acc_d = accuracy_score(ds[:,1], np.where(ds[:,0]>0.5, 1, 0))
    print(f'Diagnosis acc on 15mins: {acc_d}')
    d_pred, d_label = get_diagnose(ds)
    if abs(d_label-d_pred) < 0.5:
        print(f'Right! Diagnosis: {DIAGNOSIS[d_label]}')
    else:
        print(f'Wrong!!! Real Diagnosis: {DIAGNOSIS[d_label]}')
    return conf_mat, d_pred, d_label, ds

def get_diagnose(ds):
    label = int(ds[0,1])
    preds = ds[:,0]
    print(preds)
    pred = np.mean(preds)
    print(f'pred: {pred}, label: {label}')
    return pred, label # float, int
    
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