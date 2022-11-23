import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from config import AppConfig
from prepare import Prepare
from a_tools import myprint

savelog = 1

DEBUG_MODE = False
STANDARD_EPOCH_SEC = 30
DEFAULT_SECONDS_PER_EPOCH = 30
DEFAULT_MINUTES_PER_EPOCH = 0.5  # 30/60 or DEFAULT_SECONDS_PER_EPOCH/60;

DIAGNOSIS = ["Other","Narcolepsy type 1"]
NARCOLEPSY_PREDICTION_CUTOFF = 0.5 # if apply sigmoid (the default threshold) TODO: 阈值与ROC曲线
base = 'G:/NSRR/mnc/cnc/test/'

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
    sys.stdout = Logger('log/preprpcess_test1.txt') # 不需要自己先新建txt文档  # right: filter_right

def findAllFile(base):
    for filepath, dirnames, filenames in os.walk(base):
        for filename in filenames:
            if filename.endswith('.edf'):
                yield filepath,filename

def main(base,configInput):
    # configInput is object with additional settings.   'lightsOff','lightsOn','save','show'
    appConfig = AppConfig()
    # hyp: default
    hyp = {'show': {}, 'save': {}, 'filename': {}}
    hyp['show']['plot'] = False
    hyp['show']['hypnogram'] = False
    hyp['show']['hypnodensity'] = False
    hyp['show']['diagnosis'] = False
    hyp['save']['plot'] = True
    hyp['save']['hypnogram'] = True
    hyp['save']['hypnodensity'] = True
    hyp['save']['diagnosis'] = False
    # hyp: update with configInput
    hyp['save'].update(configInput.get('save', {}))
    hyp['show'].update(configInput.get('show', {}))

    ## For every edf file!
    for filepath,filename in findAllFile(base): # for everyone participants
        edfFilename = filepath + filename # filepath: G:/NSRR/mnc/cnc/chc/   filename: chc056-nsrr.edf
        appConfig.edf_path = edfFilename
        myprint('\nData path: ' + appConfig.edf_path)
        appConfig.xml_path = changeFileExt(edfFilename, '.xml') # sleep staging annotations

        appConfig.narcolepsy = getNCStatus(filename) # clinical diagnosis 1:NT1, 0:control
        myprint('Clinical diagnosis:', DIAGNOSIS[appConfig.narcolepsy])

        hyp['filename']['plot'] = changeFileExt(edfFilename, '.hypnodensity.png');
        hyp['filename']['hypnodensity'] = changeFileExt(edfFilename, '.hypnodensity.txt');
        hyp['filename']['hypnogram'] = changeFileExt(edfFilename, '.hypnogram.txt');
        hyp['filename']['diagnosis'] = changeFileExt(edfFilename, '.diagnosis.txt');
        hypnoConfig = hyp

        narcoApp = NarcoApp(appConfig)

        ## run the program!
        # narcoApp.eval_all()
        # signal = narcoApp.get_signal()  # get preprocessed signal self.loaded_channels
        # cdiagnosis = narcoApp.get_clinical_diagnosis() # 无需进入 prepare.py
        annotations = narcoApp.get_sleep_staging_annotation()
        print('Yes!')

        # if hypnoConfig['show']['hypnogram']:
        #     print("Hypnogram:")
        #     hypnogram = narcoApp.get_hypnogram()
        #     np.set_printoptions(threshold=10000, linewidth=150) # use linewidth = 2 to output as a single column
        #     print(hypnogram)

        # if hypnoConfig['save']['hypnogram']:
        #     narcoApp.save_hypnogram(fileName=hypnoConfig['filename']['hypnogram'])

        # if hypnoConfig['show']['hypnodensity']:
        #     print("Hypnodensity:")
        #     hypnodensity = narcoApp.get_hypnodensity()
        #     np.set_printoptions(threshold=10000*5, linewidth=150)
        #     print(hypnodensity)

        # if hypnoConfig['save']['hypnodensity']:
        #     narcoApp.save_hypnodensity(fileName=hypnoConfig['filename']['hypnodensity'])

        # if hypnoConfig['show']['diagnosis']:
        #     print(narcoApp.get_diagnosis())

        # if hypnoConfig['save']['diagnosis']:
        #     narcoApp.save_diagnosis(fileName=hypnoConfig['filename']['diagnosis'])

        # renderHypnodensity(narcoApp.get_hypnodensity(), showPlot=hypnoConfig['show']['plot'],
        #     savePlot=hypnoConfig['save']['plot'], fileName=hypnoConfig['filename']['plot'])

def getNCStatus(filename):
    # get clinical diagnosis result from filename (for CNC only)
    # filename: chc056-nsrr.edf
    if filename[2].lower() == 'c': # control
        narcolepsy = 0
    elif filename[2].lower() == 'p': # patient
        narcolepsy = 1
    return narcolepsy

def changeFileExt(fullName, newExt):
    baseName, _ = os.path.splitext(fullName)
    return baseName + newExt
    
def renderHypnodensity(hypnodensity, showPlot=False, savePlot=False, fileName='tmp.png'):
    fig, ax = plt.subplots(figsize=[11, 5])
    av = np.cumsum(hypnodensity, axis=1)
    C = [[0.90, 0.19, 0.87],  # pink
         [0.2, 0.89, 0.93],   # aqua/turquoise
         [0.22, 0.44, 0.73],  # blue
         [0.34, 0.70, 0.39]]  # green

    for i in range(4):
        xy = np.zeros([av.shape[0] * 2, 2])
        xy[:av.shape[0], 0] = np.arange(av.shape[0])
        xy[av.shape[0]:, 0] = np.flip(np.arange(av.shape[0]), axis=0)

        xy[:av.shape[0], 1] = av[:, i]
        xy[av.shape[0]:, 1] = np.flip(av[:, i + 1], axis=0)

        poly = Polygon(xy, facecolor=C[i], edgecolor=None)
        ax.add_patch(poly)

    plt.xlim([0, av.shape[0]])
    # fig.savefig('test.png')
    if savePlot:
        fig.savefig(fileName)
        # plt.savefig(fileName)

    if showPlot:
        print("Showing hypnodensity - close figure to continue.")
        plt.show()

class NarcoApp(object):

    def __init__(self, appConfig):

        # appConfig is an instance of AppConfig class, defined in inf_config.py
        self.config = appConfig
        self.edf_path = appConfig.edf_path  # full filename of an .EDF to use for header information.  A template .edf

        self.Prepare = Prepare(appConfig) # 可以把这里改成一个通用的类！
        self.narcolepsy = appConfig.narcolepsy # clinical diagnosis 1:NT1, 0:control
        # self.Hypnodensity = Hypnodensity(appConfig)

        # self.models_used = appConfig.models_used

        self.edfeatureInd = []
        self.narco_features = []
        self.narcolepsy_probability = []
        # self.extract_features = ExtractFeatures(appConfig)  <-- now in Hypnodensity

    def get_signal(self):
        return self.Prepare.get_signal()

    def get_clinical_diagnosis(self):
        return self.narcolepsy # clinical diagnosis 1:NT1, 0:control

    def get_sleep_staging_annotation(self):
        return self.Prepare.get_annotation()

    def get_diagnosis(self):
        prediction = self.narcolepsy_probability
        if not prediction:
            prediction = self.get_narco_prediction()
        return "Score: %0.4f\nDiagnosis: %s"%(prediction[0],DIAGNOSIS[int(prediction>=NARCOLEPSY_PREDICTION_CUTOFF)])

    def get_hypnodensity(self):
        return self.Hypnodensity.get_hypnodensity()

    def get_hypnogram(self):
        return self.Hypnodensity.get_hypnogram()

    def save_diagnosis(self, fileName=''):
        if fileName == '':
            fileName = changeFileExt(self.edf_path, '.diagnosis.txt')
        with open(fileName,"w") as textFile:
            print(self.get_diagnosis(),file=textFile)

    def save_hypnodensity(self, fileName=''):
        if fileName == '':
            fileName = changeFileExt(self.edf_path, '.hypnodensity.txt')
        hypno = self.get_hypnodensity()
        np.savetxt(fileName, hypno, delimiter=",")

    def save_hypnogram(self, fileName=''):
        if fileName == '':
            fileName = changeFileExt(self.edf_path, '.hypnogram.txt')

        hypno = self.get_hypnogram()
        np.savetxt(fileName, hypno, delimiter=",", fmt='%i')

    def get_narco_gpmodels(self):

        return self.models_used

    def get_hypnodensity_features(self, modelName, idx):
        return self.Hypnodensity.get_features(modelName, idx)

    def get_narco_prediction(self):  # ,current_subset, num_subjects, num_models, num_folds):
        pass
        # # Initialize dummy variables
        # num_subjects = 1
        # gpmodels = self.get_narco_gpmodels()
        # num_models = len(gpmodels)
        # num_folds = self.config.narco_prediction_num_folds

        # mean_pred = np.zeros([num_subjects, num_models, num_folds])
        # var_pred = np.zeros([num_subjects, num_models, num_folds])

        # scales = self.config.narco_prediction_scales
        # gpmodels_base_path = self.config.narco_classifier_path

        # for idx, gpmodel in enumerate(gpmodels):
        #     print('{} | Predicting using: {}'.format(datetime.now(), gpmodel))

        #     X = self.get_hypnodensity_features(gpmodel, idx)

        #     for k in range(num_folds):
        #         #         print('{} | Loading and predicting using {}'.format(datetime.now(), os.path.join(gpmodels_base_path, gpmodel, gpmodel + '_fold{:02}.gpm'.format(k+1))))
        #         with tf.Graph().as_default() as graph:
        #             with tf.Session():
        #                 path_gp = os.path.join(gpmodels_base_path, gpmodel, gpmodel + '_fold{:02}.gpm'.format(k + 1))
        #                 print(path_gp)
        #                 gpfSave = gpf.saver.Saver()
        #                 m = gpfSave.load(path_gp) # bug! here!
        #                 mean_pred[:, idx, k, np.newaxis], var_pred[:, idx, k, np.newaxis] = m.predict_y(X)

        # self.narcolepsy_probability = np.sum(np.multiply(np.mean(mean_pred, axis=2), scales), axis=1) / np.sum(scales)
        # return self.narcolepsy_probability

    def plotHypnodensity(self):
        self.Hypnodensity.renderHynodenisty(option="plot")

    def eval_hypnodensity(self):
        self.Hypnodensity.evaluate()

    def eval_narcolepsy(self):
        self.get_narco_prediction()

    def eval_all(self):
        self.eval_hypnodensity()
        self.eval_narcolepsy()


if __name__ == '__main__':
    jsonObj = json.loads('{"show":{"plot":true,"hypnodensity":false,"hypnogram":false}, "save":{"plot":true,"hypnodensity":true, "hypnogram":true}}')
    try:
        main(base, jsonObj)
    except OSError as oserr:
        print("OSError:", oserr)