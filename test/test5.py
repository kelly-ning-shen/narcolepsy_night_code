import pickle
from pathlib import Path
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


DURATION_MINUTES = 15
DEFAULT_MINUTES_PER_EPOCH = 0.5
DEFAULT_SECONDS_PER_EPOCH = 30
fs = 100

nepoch = int(DURATION_MINUTES/DEFAULT_MINUTES_PER_EPOCH)
nsample = int(nepoch*DEFAULT_SECONDS_PER_EPOCH*fs)


def viewasIMG(EEG,EOG,EMG,ann,i):
    EEG = signaltoSquare(EEG)
    EOG = signaltoSquare(EOG)
    EMG = signaltoSquare(EMG)
    signal_pic = np.array([EEG,EOG,EMG]) # TODO: 如何知道这是不是对应上了RGB这样的三层
    signal_pic = signal_pic.transpose(1,2,0) # 第一层是否是指EEG，第二层是否是指EOG，第三层是否是指EMG
    print(f'Shape of signal_pic: {signal_pic.shape}')
    plt.imshow(signal_pic)
    tickpos = np.arange(nepoch)*(signal_pic.shape[0]/nepoch)
    plt.yticks(tickpos,ann)
    # plt.colorbar()
    plt.savefig(f'pic/chp001-nsrr_15min_{i}_ann.png', bbox_inches='tight')
    plt.close()
    # plt.show()
    # input()

def signaltoSquare(signal):
    # # 1. min-max normalization (0-1)
    # tool = MinMaxScaler(feature_range=(0,1)) # min:0, max:1
    # signal = tool.fit_transform(signal[:, np.newaxis])
    # 2. reshape to square
    lens = int(np.sqrt(len(signal)))
    signal = signal.reshape(lens,lens)
    return signal
    
if __name__ == '__main__':
    p = Path('data/mnc/cnc/chp/chp001-nsrr.pkl')
    a = Path('data/mnc/cnc/chp/chp001-nsrr.ann_pkl')
    with p.open('rb') as fp:
        loaded_channels = pickle.load(fp)
    with a.open('rb') as fp:
        annotations = pickle.load(fp)

    C3 = loaded_channels['C3']
    C4 = loaded_channels['C4']
    E1 = loaded_channels['E1']
    E2 = loaded_channels['E2']
    EMG = loaded_channels['EMG']
    nduration = len(C4) // nsample
    print('Finish loading!')

    # 1. min-max normalization (0-1)
    tool = MinMaxScaler(feature_range=(0,1)) # min:0, max:1
    C4 = tool.fit_transform(C4[:, np.newaxis])
    E1 = tool.fit_transform(E1[:, np.newaxis])
    EMG = tool.fit_transform(EMG[:, np.newaxis])

    for i in range(nduration):
        EEG1 = C4[i*nsample:(i+1)*nsample]
        EOG1 = E1[i*nsample:(i+1)*nsample]
        EMG1 = EMG[i*nsample:(i+1)*nsample]
        ann1 = annotations[i*nepoch:(i+1)*nepoch]
    
        viewasIMG(EEG1,EOG1,EMG1,ann1,i)