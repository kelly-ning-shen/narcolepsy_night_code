import pickle
from pathlib import Path

base = 'data/mnc/cnc/cnc/'
DURATION_MINUTES = 0.5 # my first choice: 15min

def loadSubjectData(base):
    '''
    return filepaths of train and test
    '''
    subjects = sorted(Path(base).glob(f'*.xml'))
    nsubject = len(subjects)
    print(f'Subjects num: {nsubject}\n')

    subjects_data = {}
    n15minduration = 0
    if DURATION_MINUTES % 1 == 0:
        idx = 0
    else:
        idx = 1
    # read: the inputs of one subject (save to dict)
    for i in range(nsubject):
        subject = subjects[i] # WindowsPath('dsata/mnc/cnc/cnc/chc001-nsrr.xml')
        name = subject.stem
        subject_data = sorted(Path(base).glob(f'{name}_{DURATION_MINUTES}min_zscore_*.s_pkl'))
        subject_data = sorted(subject_data, key=lambda x: int(str(x).split('.')[idx].split('_')[-1]))
        ndata = len(subject_data)
        print(f'{DURATION_MINUTES}min inputs: {name} ({ndata})')
        subjects_data[name] = subject_data # list of datapath
        n15minduration += len(subject_data)
    
    return subjects_data, n15minduration

subjects_data, n15minduration = loadSubjectData(base)