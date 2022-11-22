import os
import numpy as np


class AppConfig(object):

    def __init__(self):

        # # Model folder
        # self.models_used = ['ac_rh_ls_lstm_01', 'ac_rh_ls_lstm_02',
        #                     'ac_rh_ls_lstm_03', 'ac_rh_ls_lstm_04',
        #                     'ac_rh_ls_lstm_05', 'ac_rh_ls_lstm_06',
        #                     'ac_rh_ls_lstm_07', 'ac_rh_ls_lstm_08',
        #                     'ac_rh_ls_lstm_09', 'ac_rh_ls_lstm_10',
        #                     'ac_rh_ls_lstm_11', 'ac_rh_ls_lstm_12',
        #                     'ac_rh_ls_lstm_13', 'ac_rh_ls_lstm_14',
        #                     'ac_rh_ls_lstm_15', 'ac_rh_ls_lstm_16']

        # # Uncomment the following when running validation comparison given in readme file.
        # # self.models_used = ['ac_rh_ls_lstm_01']

        # Hypnodensity classification settings
        # self.relevance_threshold = 1 # 没有被使用！
        self.fs = np.array(100,dtype=float)
        self.fsH = np.array(0.2,dtype=float)
        self.fsL = np.array(49,dtype=float)

        self.channels = ['C3','C4','O1','O2','E1','E2','EMG']

        # # Size of cross correlation in seconds - so in samples this will be sum([200 200 400 400 40 ]) == 1240 + 400 for EOGLR == 1640
        # self.CCsize = {'C3':   2, 'C4':   2,  'O1':   2, 'O2':   2,
        #                'E1':4, 'E2':4,
        #                'EMG':  0.4
        #                }
        # #self.CCsize = dict(zip(self.channels,
        # #                [2,2,2,2,4,4,0.4]))
        self.channels_used = dict.fromkeys(self.channels) # {'C3': None, 'C4': None, 'O1': None, 'O2': None, 'E1': None, 'E1': None, 'EMG': None}
        self.loaded_channels = dict.fromkeys(self.channels)

        # self.psg_noise_file_pathname = './ml/noiseM.mat'
        # self.hypnodensity_model_root_path = './ml/'
        # self.hypnodensity_scale_path = './ml/scaling/'
        # # self.hypnodensity_select_features_path = './ml/'
        # # self.hypnodensity_select_features_pickle_name = 'narcoFeatureSelect.p'

        self.Kfold = 10  # or 20
        self.edf_path = []
        self.lightsOff = []
        self.lightsOn = []

        # # Related to classifying narcolepsy from hypnodensity features
        # self.narco_classifier_path = '.\\ml\\gp'

        # self.narco_prediction_num_folds = 5 # for the gp narco classifier
        # self.narco_prediction_scales = [0.90403101, 0.89939177, 0.90552177, 0.88393560,
        #   0.89625522, 0.88085868, 0.89474061, 0.87774597,
        #   0.87615981, 0.88391175, 0.89158020, 0.88084675,
        #   0.89320215, 0.87923673, 0.87615981, 0.88850328]

        # self.narco_prediction_selected_features = [1, 11, 16, 22, 25, 41, 43, 49, 64, 65, 86, 87, 103, 119, 140,
        #                                            147, 149, 166, 196, 201, 202, 220, 244, 245, 261, 276, 289, 296,
        #                                            299, 390, 405, 450, 467, 468, 470, 474, 476, 477] # 诊断环节所选的38个特征的编号

