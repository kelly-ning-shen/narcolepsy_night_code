# find channel index in edf
# 放到预处理部分！
# -*- coding: utf-8 -*-
"""
python 3.6.7 ('stanford-stages': conda)
Using pyedflib to read signal information.
"""
import pyedflib

channels = ['C3','C4','O1','O2','E1','E2','EMG']
fullname = 'G:/NSRR/mnc/cnc/chp/chp040-nsrr.edf'
channels_used = dict.fromkeys(channels) # record channel index (start from 0)

f = pyedflib.EdfReader(fullname)

signal_labels = f.getSignalLabels() # list
print('\nlabels: ', signal_labels)

for ch in channels:
    if ch != 'EMG':
        channels_used[ch] = signal_labels.index(ch)
    else:
        emglabels = ('cchin_l', 'chin')
        for e in emglabels:
            if e in signal_labels:
                emglabel = e
                print('The real label of EMG:', emglabel)
        # if emglabels[0] in signal_labels:
        #     emglabel = emglabels[0]
        # else:
        #     emglabel = emglabels[1]
        channels_used[ch] = signal_labels.index(emglabel)

print(channels_used)
