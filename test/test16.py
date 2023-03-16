import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch import optim
# # from torch.autograd import Function
# from torch.utils.data import Dataset, DataLoader
# import torchvision

# from network import SquareSmall10min
from network import MultiCNNC2CM
# from a_tools import myprint
# from a_metrics import plot_confusion_matrix, plot_ROC_curve

savelog = 1
savepic = 1
savecheckpoints = 1

is_multitask = 1 # 0: onephase, 1: multitask
is_per_epoch = 1 # input: 0: sqauresmall, 1: multitask
DURATION_MINUTES = 2.5 # my first choice: 15min
DEFAULT_MINUTES_PER_EPOCH = 0.5  # 30/60 or DEFAULT_SECONDS_PER_EPOCH/60;
nepoch = int(DURATION_MINUTES/DEFAULT_MINUTES_PER_EPOCH)

subject = 'chc008-nsrr'

model = MultiCNNC2CM(n_channels=3,nepoch=nepoch)
# if torch.cuda.device_count()>1:
#     model = nn.DataParallel(model)
# model = nn.DataParallel(model, device_ids=[0,1])
PATH = f'checkpoints/multicnnc2cm_{DURATION_MINUTES}min_zscore_shuffle_ROC/checkpoints_{subject}.pth'
model.load_state_dict(torch.load(PATH))
# model.to(device)