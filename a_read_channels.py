# -*- coding: utf-8 -*-
"""
python 3.7.1 64-bit (global)
"""
from mne.io import read_raw_edf
import matplotlib.pyplot as plt
# import mne
import pandas as pd
import os

base = 'G:/NSRR/mnc/cnc/chp/'



def findAllFile(base):
    for filepath, dirnames, filenames in os.walk(base):
        for filename in filenames:
            if filename.endswith('.edf'):
                yield filepath,filename

def readSignalInfo(base,save=True):
    total = 25
    chs_names_pad = []
    subject = []
    sample_Hz = []
    for filepath,filename in findAllFile(base):
        fullname = filepath + filename # filepath: G:/NSRR/mnc/cnc/chc/   filename: chc056-nsrr.edf
        print('\ndata path: ' + fullname)

        print('Next step: read channels')
        try:
            raw = read_raw_edf(fullname, preload=False)
        except RuntimeWarning:
            pass
        else:
            ch_names = raw.info['ch_names']
            s_Hz = raw.info['sfreq']
            ch_names_pad = ch_names + [' ']*(total-len(ch_names)) # total: 要取最大的通道数，这样才能形成一个矩阵
            chs_names_pad.append(ch_names_pad)
            subject.append(filename)
            sample_Hz.append(s_Hz)

    # After searching for all subjects
    chs_names_dict = dict(zip(subject,chs_names_pad))
    chs_names_dt = pd.DataFrame(chs_names_dict)
    print(chs_names_dt)
    print(sample_Hz)

    if save:
        savetoExcel(chs_names_dt)

def savetoExcel(chs_names_dt):
    # save dataframe to excel
    writer = pd.ExcelWriter('log/channel/cnc_chp.xlsx') # pylint: disable=abstract-class-instantiated
    chs_names_dt.to_excel(writer)
    writer.save()
    print('DataFrame is written successfully to the Excel File.')


if __name__ == '__main__':
    readSignalInfo(base,save=False)