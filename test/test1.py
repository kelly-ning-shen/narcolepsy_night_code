# -*- coding: utf-8 -*-
"""
python 3.6.7 ('stanford-stages': conda)
Using pyedflib to read signal information.
"""
import pyedflib

def readSignalInfo():
    fullname = 'G:/NSRR/mnc/cnc/chp/chp040-nsrr.edf'

    f = pyedflib.EdfReader(fullname)

    n = f.signals_in_file
    print('\nsignal numbers: ', n)

    signal_labels = f.getSignalLabels() # list
    print('\nlabels: ', signal_labels)

    signal_fs = f.getSampleFrequencies() # array
    print('\nfs: ', signal_fs)

    # concat 2 list with ': '
    for i in range(n):
        signal_labels[i] += ': ' + str(signal_fs[i])
    print(signal_labels)

    # Test2: pyedflib - EdfReader - readSignal(chn)
    # chn, namely channel number: start from 0, or 1?
    for i in range(n):
        print('\n======{} {}====='.format(i,signal_labels[i]))
        signal = f.readSignal(i)
        print('signal first 5 points: ', signal[0:5])
        print('signal last 5 points: ', signal[-5:])


    # header = f.getHeader()  # {'technician': '', 'recording_additional': '', 'patientname': '', 'patient_additional': '', 'patientcode': '', 'equipment': '', 'admincode': '', 
    #                         # 'gender': '', 'startdate': datetime.datetime(1985, 1, 1, 20, 2, 42), 'birthdate': ''}
    # print(header)

    # signal_headers = f.getSignalHeaders() # output: headers about all signals (label, sample_rate,physical_max ...)
    # print(signal_headers)

    f._close()
    del f

if __name__ == '__main__':
    readSignalInfo()