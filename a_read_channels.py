# -*- coding: utf-8 -*-
"""
python 3.6.7 ('stanford-stages': conda)
Using pyedflib to read signal information.
"""
import pyedflib
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
    for filepath,filename in findAllFile(base):
        fullname = filepath + filename # filepath: G:/NSRR/mnc/cnc/chc/   filename: chc056-nsrr.edf
        print('\ndata path: ' + fullname)

        print('Next step: read channels')
        try:
            f = pyedflib.EdfReader(fullname)
        except RuntimeWarning:
            pass
        else:
            n = f.signals_in_file # signal numbers
            signal_labels = f.getSignalLabels() # list
            signal_fs = f.getSampleFrequencies() # array

            # concat 2 list with ': '
            for i in range(n):
                signal_labels[i] += ': ' + str(signal_fs[i])
            # print(signal_labels)

            ch_names_pad = signal_labels + [' ']*(total-len(signal_labels)) # total: 要取最大的通道数，这样才能形成一个矩阵
            chs_names_pad.append(ch_names_pad)
            subject.append(filename.split('.')[0]) # w/o .edf

            f._close()
            del f

    # After searching for all subjects
    chs_names_dict = dict(zip(subject,chs_names_pad))
    chs_names_dt = pd.DataFrame(chs_names_dict)
    print(chs_names_dt)


    if save:
        savetoExcel(chs_names_dt)

def savetoExcel(chs_names_dt):
    # save dataframe to excel
    writer = pd.ExcelWriter('log/channel/cnc_chp.xlsx') # pylint: disable=abstract-class-instantiated
    chs_names_dt.to_excel(writer)
    writer.save()
    print('DataFrame is written successfully to the Excel File.')


if __name__ == '__main__':
    readSignalInfo(base,save=True)