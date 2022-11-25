# -*- coding: utf-8 -*-
"""
prepare:
1. PreProcess (class)
2. main part
    2.1 OnePhase (class)
    2.2 TwoPhase (class)
    2.3 MultiTask (class)
"""
import os
import pickle
from pathlib import Path

import numpy as np
import pyedflib
import scipy.signal as signal  # for edf channel sampling and filtering

from a_tools import myprint

def softmax(x):
    e_x = np.exp(x)
    div = np.repeat(np.expand_dims(np.sum(e_x, axis=1), 1), 5, axis=1)
    return np.divide(e_x, div)

class PreProcess(object):
    
    def __init__(self,appConfig):
        self.config = appConfig

        self.channels = appConfig.channels
        self.channels_used = appConfig.channels_used # dict 记录每个通道是否被使用 {'C3': 3, 'C4': 4, 'O1': 5, 'O2': 6, 'EOG-L': 7, 'EOG-R': 8, 'EMG': 9, 'A1': None, 'A2': None}
        self.loaded_channels = appConfig.loaded_channels # dict 收集所需的通道的信号（单位：uV）
        self.edf_pathname = appConfig.edf_path
        # self.encodedD = []
        self.fs = int(appConfig.fs) # 100 Hz
        self.fsH = appConfig.fsH
        self.fsL = appConfig.fsL
        self.lightsOff = appConfig.lightsOff
        self.lightsOn = appConfig.lightsOn

        self.edf = []  # pyedflib.EdfFileReader

    def preprocessing(self):
        p = Path(self.edf_pathname) # PosixPath('CHP040') # 这里修改保存的位置
        p = Path(p.with_suffix('.pkl')) # PosixPath('CHP040.pkl')

        if (p.exists()):

            myprint('Loading previously saved preprocessed data')
            with p.open('rb') as fp:
                self.loaded_channels = pickle.load(fp)
            # myprint('PAUSE HERE!')
        else:
            myprint('Load EDF')
            self.loadEDF() # 2022/11/22: finish

            # myprint('Load noise level')
            # self.psg_noise_level()

            self.filtering() # 0.2Hz-49Hz
            myprint('filtering done')

            # myprint('Encode')
            # self.encoding()

            # pickle our file
            with p.open('wb') as fp:
                pickle.dump(self.loaded_channels, fp)
                myprint("pickling done")
        return self.loaded_channels
    
    def channelUsed(self):
        # get the index of used channels
        signal_labels = self.edf.getSignalLabels() # list
        myprint(f'original signals ({len(signal_labels)}): {signal_labels}')

        for ch in self.channels:
            if ch != 'EMG':
                self.channels_used[ch] = signal_labels.index(ch)
            else:
                emglabels = ('cchin_l', 'chin')
                for e in emglabels:
                    if e in signal_labels:
                        emglabel = e
                        myprint('The real label of EMG:', emglabel)
                self.channels_used[ch] = signal_labels.index(emglabel)

    def loadEDF(self):
        if not self.edf:
            try:
                self.edf = pyedflib.EdfReader(self.edf_pathname)
            except OSError as osErr:
                print('OSError:', 'Loading', self.edf_pathname)
                raise(osErr)
        self.channelUsed() # update self.channels_used
        for ch in self.channels:  # self.channels = ['C3','C4','O1','O2','E1','E2','EMG']
            myprint('Loading', ch)
            if isinstance(self.channels_used[ch], int):
                myprint(self.channels_used[ch], ch)
                self.loaded_channels[ch] = self.edf.readSignal(self.channels_used[ch]) # read signals according to channels index
                if self.edf.getPhysicalDimension(self.channels_used[ch]).lower() == 'mv': # signal的单位统一成uV
                    myprint('mv')
                    self.loaded_channels[ch] *= 1e3
                elif self.edf.getPhysicalDimension(self.channels_used[ch]).lower() == 'v':
                    myprint('v')
                    self.loaded_channels[ch] *= 1e6

                fs = int(self.edf.samplefrequency(self.channels_used[ch]))
                # fs = Decimal(fs).quantize(Decimal('.0001'), rounding=ROUND_DOWN)
                myprint('fs', fs)

                self.resampling(ch, fs)
                myprint('Resampling done')

                # Trim excess
                self.trim(ch)

            else:
                print('channel[', ch, '] was empty (skipped)', sep='') # 如果采集的信号中不包含这个通道，那就在dict中删去
                del self.channels_used[ch]

    def trim(self, ch):
        # 30 represents the epoch length most often used in standard hypnogram scoring.
        rem = len(self.loaded_channels[ch]) % int(self.fs * 30)
        # Otherwise, if rem == 0, the following results in an empty array
        if rem>0: # 删掉最后不满足一个epoch长度的部分
            self.loaded_channels[ch] = self.loaded_channels[ch][:-rem]

    def resampling(self, ch, fs):
        myprint("original samplerate = ", fs)
        myprint("resampling to ", self.fs)
        if fs == 500 or fs == 200:
            numerator = [[-0.0175636017706537, -0.0208207236911009, -0.0186368912579407, 0.0, 0.0376532652007562,
                0.0894912177899215, 0.143586518157187, 0.184663795586300, 0.200000000000000, 0.184663795586300,
                0.143586518157187, 0.0894912177899215, 0.0376532652007562, 0.0, -0.0186368912579407,
                -0.0208207236911009, -0.0175636017706537],
                [-0.050624178425469, 0.0, 0.295059334702992, 0.500000000000000, 0.295059334702992, 0.0,
                -0.050624178425469]]  # from matlab
            if fs == 500:
                s = signal.dlti(numerator[0], [1], dt=1. / self.fs)
                self.loaded_channels[ch] = signal.decimate(self.loaded_channels[ch], fs // self.fs, ftype=s, zero_phase=False)
            elif fs == 200:
                s = signal.dlti(numerator[1], [1], dt=1. / self.fs)
                self.loaded_channels[ch] = signal.decimate(self.loaded_channels[ch], fs // self.fs, ftype=s, zero_phase=False)
        else:
            self.loaded_channels[ch] = signal.resample_poly(self.loaded_channels[ch],
                                                            self.fs, fs, axis=0, window=('kaiser', 5.0))

    def filtering(self):
        myprint('Filtering remaining signals')
        fs = self.fs # 100Hz

        Fh = signal.butter(5, self.fsH / (fs / 2), btype='highpass', output='ba') # 0.2Hz
        Fl = signal.butter(5, self.fsL / (fs / 2), btype='lowpass', output='ba') # 49Hz

        for ch, ch_idx in self.channels_used.items():
            # Fix for issue 9: https://github.com/Stanford-STAGES/stanford-stages/issues/9
            if isinstance(ch_idx, int):
                myprint('Filtering {}'.format(ch))
                self.loaded_channels[ch] = signal.filtfilt(Fh[0], Fh[1], self.loaded_channels[ch])

                if fs > (2 * self.fsL):
                    self.loaded_channels[ch] = signal.filtfilt(Fl[0], Fl[1], self.loaded_channels[ch]).astype(
                        dtype=np.float32)


class Prepare(object):
    def __init__(self,appConfig):
        self.config = appConfig
        self.fs = appConfig.fs
        self.PreProcessing = PreProcess(appConfig)
        self.xml_path = appConfig.xml_path
        # self.narcolepsy = appConfig.narcolepsy
    
    def preparing(self):
        myprint('Get signal')
        self.get_signal()

        myprint('Get annotations')
        self.get_annotation()

        myprint('Check epochs num...')
        self.check_epochs()

        try:
            return self.loaded_channels, self.annotations, self.erridx
        except AttributeError:
            return self.loaded_channels, self.annotations

    def get_signal(self):
        # get preprocessed signal (without trim misssing anns)
        self.loaded_channels = self.PreProcessing.preprocessing()
        self.channels = list(self.loaded_channels.keys())
        myprint(f'Load preprocessed signal for diagnosis ({len(self.channels)}): {self.channels}')
        return self.loaded_channels

    # def get_diagnosis(self):
    #     # get clinical diagnosis (myprint in MAIN)
    #     return self.narcolepsy
        
    def get_annotation(self):
        # get sleep staging annotation (without trim misssing anns)
        a = Path(self.xml_path)
        a = Path(a.with_suffix('.ann_pkl'))

        if (a.exists()):
            myprint('Loading previously read sleep staging annotations')
            with a.open('rb') as fp:
                self.annotations = pickle.load(fp)
        else:
            myprint('Read annotations from xml')
            self.annotations = self.readAnnfromXml()

            # pickle our file
            with a.open('wb') as fp:
                pickle.dump(self.annotations, fp)
                myprint('pickling done')

        return self.annotations

    def readAnnfromXml(self):
        annotation_dict = {
            'wake': 0,
            'NREM1': 1,
            'NREM2': 2,
            'NREM3': 3,
            'REM': 4
        }
        annotations = []
        start = -1
        import xml.etree.ElementTree as ET
        ScoredEvents = ET.parse(self.xml_path).find('Instances').findall('Instance')
        for event in ScoredEvents:
            # event_type = event.find('EventType').text
            # if event_type != 'Stages|Stages':
            #     continue
            event_concept = event.attrib['class']
            # start_time = float(event.find('Start').text)
            duration = float(event.find('Duration').text)
            assert duration % 30 == 0
            N = int(duration // 30) # if not 30s one annotation
            if event_concept in annotation_dict.keys():
                anns = [annotation_dict[event_concept]] * N
            else:
                anns = [-1] * N
            if start == -1:
                start = float(event.find('Start').text)
            annotations.extend(anns)
        # return start, annotations # start 记录最开始的时间点 (all start from 0)
        return np.array(annotations)
        
    def check_epochs(self):
        epoch_num_signal = len(self.loaded_channels[self.channels[0]])/int(self.fs * 30)
        epoch_num_ann = len(self.annotations)
        if epoch_num_signal != epoch_num_ann:
            print(f'Different epochs num: signal ({epoch_num_signal}), ann ({epoch_num_ann})')
        else:
            print(f'Same epochs num: {len(self.annotations)}')
            self.trim_epochs()

    def trim_epochs(self):
        # if same epoch num in signal and anns
        erridx = np.where(self.annotations==-1)[0]
        if len(erridx)>0 and (erridx[0] == 0 or erridx[-1] == len(self.annotations)-1):
            erridx_del = erridx # 默认都在头或尾
            if erridx.size != erridx[-1]-erridx[0]+1:
                # 如果中间还有缺失的
                # 只删去开头或结尾的（需要找到连续的idx）
                # 找到[0,1,2,3]
                # 这里不存在头和尾都异常的
                erridx_d = np.diff(erridx)
                idx = np.where(erridx_d==1)[0] # idx: index of erridx_d
                erridx_del = erridx[idx[0]:idx[-1]+2] # 需要删除的erridx
            erridx = np.setdiff1d(erridx,erridx_del) # which missing values are still there

            myprint('Delete missing value in anns')
            self.annotations = np.delete(self.annotations,erridx_del)
            print(f'Trim epochs to {len(self.annotations)}')

            myprint('Delete the corresponding epochs in signal')
            start = int(self.fs * 30 * erridx_del[0])
            finish = int(self.fs * 30 * (erridx_del[-1]+1))
            for ch in self.channels:
                # self.loaded_channels[ch] = self.loaded_channels[ch][start:finish] # 这是要截掉的部分！不是要保留的部分！
                self.loaded_channels[ch] = np.delete(self.loaded_channels[ch], np.arange(start,finish))
                print(len(self.loaded_channels[ch])/3000)

        if len(erridx) > 0:
            self.erridx = erridx # which missing values are still there