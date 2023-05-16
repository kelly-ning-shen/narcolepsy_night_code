import torch
import torch.nn as nn
import numpy as np

####################### MODEL COMPONENTS #######################
class SingleConv(nn.Module):
    '''[kernel: square] (conv -> BN -> ReLU)'''
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(SingleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class OutSleepStage(nn.Module):
    '''conv [kernel: 1*1]*out_channels)'''
    def __init__(self, in_channels, out_channels):
        super(OutSleepStage, self).__init__()
        self.outsleepstage = nn.Conv2d(in_channels,out_channels,kernel_size=1,padding=0)
    def forward(self, x):
        x = self.outsleepstage(x)
        return x

class OutDiagnosis(nn.Module):
    '''(flatten -> fc -> fc)'''
    def __init__(self, in_channels, hidden_channels):
        # in_channels: the number of features that are input to the fully-connected layer after flattening
        super(OutDiagnosis, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, 1)
        )
    def forward(self, x):
        x = self.fc(x)
        return x

class Expert_net(nn.Module):
    '''squaresmall, 2.5 min
    torch.Size([10, 3, 100, 150]) → torch.Size([10, 128, 5, 5])
    out: (bs, out_channels, nepoch, nepoch)'''
    def __init__(self,in_channels):
        super(Expert_net, self).__init__()
        
        self.cnn_layer1 = SingleConv(in_channels, 64, kernel_size=(3,5), stride=(2,3), padding=1)     # main: conv2d + batchnorm + relu
        self.cnn_layer2 = SingleConv(64, 128, kernel_size=5, stride=3, padding=0)            # main: conv2d + batchnorm + relu
        self.cnn_layer3 = SingleConv(128, 128, kernel_size=4, stride=3, padding=0)
        

    def forward(self, x):
        x = self.cnn_layer1(x)
        x = self.cnn_layer2(x)
        out = self.cnn_layer3(x)
        return out

class Extraction_Network(nn.Module):
    '''FeatureDim-输入数据的维数; ExpertOutDim-每个Expert输出的维数; TaskExpertNum-任务特定专家数;
       CommonExpertNum-共享专家数; GateNum-gate数(2表示最后一层, 3表示中间层)'''
    def __init__(self,in_channels,FeatureDim,TaskExpertNum,CommonExpertNum,GateNum): 
        super(Extraction_Network, self).__init__()
        
        self.GateNum = GateNum #输出几个Gate的结果，2表示最后一层只输出两个任务的Gate，3表示还要输出中间共享层的Gate
        
        '''两个任务模块，一个共享模块'''
        self.n_task = 2
        self.n_share = 1
        
        '''TaskA-Experts'''
        for i in range(TaskExpertNum):
            setattr(self, "expert_layer"+str(i+1), Expert_net(in_channels)) 
        self.Experts_A = [getattr(self,"expert_layer"+str(i+1)).cuda() for i in range(TaskExpertNum)]#Experts_A模块，TaskExpertNum个Expert
        '''Shared-Experts'''
        for i in range(CommonExpertNum):
            setattr(self, "expert_layer"+str(i+1), Expert_net(in_channels)) 
        self.Experts_Shared = [getattr(self,"expert_layer"+str(i+1)).cuda() for i in range(CommonExpertNum)]#Experts_Shared模块，CommonExpertNum个Expert
        '''TaskB-Experts'''
        for i in range(TaskExpertNum):
            setattr(self, "expert_layer"+str(i+1), Expert_net(in_channels)) 
        self.Experts_B = [getattr(self,"expert_layer"+str(i+1)).cuda() for i in range(TaskExpertNum)]#Experts_B模块，TaskExpertNum个Expert
        
        '''Task_Gate网络结构'''
        for i in range(self.n_task):
            setattr(self, "gate_layer"+str(i+1), nn.Sequential(nn.Flatten(),
                                                                nn.Linear(FeatureDim, TaskExpertNum+CommonExpertNum),
                                        					    nn.Softmax(dim=1))) 
        self.Task_Gates = [getattr(self,"gate_layer"+str(i+1)).cuda() for i in range(self.n_task)]#为每个gate创建一个lr+softmax      
        '''Shared_Gate网络结构'''
        for i in range(self.n_share):
            setattr(self, "gate_layer"+str(i+1), nn.Sequential(nn.Flatten(),
                                                                nn.Linear(FeatureDim, 2*TaskExpertNum+CommonExpertNum),
                                        					    nn.Softmax(dim=1))) 
        self.Shared_Gates = [getattr(self,"gate_layer"+str(i+1)).cuda() for i in range(self.n_share)]#共享gate       
        
    def forward(self, x_A, x_S, x_B):
        
        '''Experts_A模块输出'''
        Experts_A_Out = [expert(x_A) for expert in self.Experts_A] #
        ExpertOutDim = Experts_A_Out[0].shape
        # print(ExpertOutDim)
        Experts_A_Out = torch.cat(([expert[:,np.newaxis,:,:,:] for expert in Experts_A_Out]),dim = 1) # 维度 (bs,TaskExpertNum,ExpertOutDim)
        # print(Experts_A_Out.shape)
        '''Experts_Shared模块输出'''
        Experts_Shared_Out = [expert(x_S) for expert in self.Experts_Shared] #
        Experts_Shared_Out = torch.cat(([expert[:,np.newaxis,:,:,:] for expert in Experts_Shared_Out]),dim = 1) # 维度 (bs,CommonExpertNum,ExpertOutDim)
        '''Experts_B模块输出'''
        Experts_B_Out = [expert(x_B) for expert in self.Experts_B] #
        Experts_B_Out = torch.cat(([expert[:,np.newaxis,:,:,:] for expert in Experts_B_Out]),dim = 1) # 维度 (bs,TaskExpertNum,ExpertOutDim)
        
        '''Gate_A的权重'''
        Gate_A = self.Task_Gates[0](x_A)     # 维度 n_task个(bs,TaskExpertNum+CommonExpertNum)
        # '''Gate_Shared的权重'''
        # if self.GateNum == 3:
        #     Gate_Shared = self.Shared_Gates[0](x_S)     # 维度 n_task个(bs,2*TaskExpertNum+CommonExpertNum)
        '''Gate_B的权重'''
        Gate_B = self.Task_Gates[1](x_B)     # 维度 n_task个(bs,TaskExpertNum+CommonExpertNum)
             
        '''GateA输出'''
        g = Gate_A.unsqueeze(2)  # 维度(bs,TaskExpertNum+CommonExpertNum,1)
        experts = torch.cat([Experts_A_Out,Experts_Shared_Out],dim=1) #维度(bs,TaskExpertNum+CommonExpertNum,ExpertOutDim)
        bs = g.shape[0]
        TotalExpertNum = experts.shape[1]
        experts = torch.reshape(experts, (bs,TotalExpertNum,-1))
        Gate_A_Out = torch.matmul(experts.transpose(1,2),g)#维度(bs,ExpertOutDim,1)
        Gate_A_Out = Gate_A_Out.squeeze(2)#维度(bs,ExpertOutDim)  
        Gate_A_Out = torch.reshape(Gate_A_Out, ExpertOutDim)
        # '''GateShared输出'''
        # if self.GateNum == 3:
        #     g = Gate_Shared.unsqueeze(2)  # 维度(bs,2*TaskExpertNum+CommonExpertNum,1)
        #     experts = torch.cat([Experts_A_Out,Experts_Shared_Out,Experts_B_Out],dim=1) #维度(bs,2*TaskExpertNum+CommonExpertNum,ExpertOutDim)
        #     Gate_Shared_Out = torch.matmul(experts.transpose(1,2),g)#维度(bs,ExpertOutDim,1)
        #     Gate_Shared_Out = Gate_Shared_Out.squeeze(2)#维度(bs,ExpertOutDim)        
        '''GateB输出'''
        g = Gate_B.unsqueeze(2)  # 维度(bs,TaskExpertNum+CommonExpertNum,1)
        experts = torch.cat([Experts_B_Out,Experts_Shared_Out],dim=1) #维度(bs,TaskExpertNum+CommonExpertNum,ExpertOutDim)
        experts = torch.reshape(experts, (bs,TotalExpertNum,-1))
        Gate_B_Out = torch.matmul(experts.transpose(1,2),g)#维度(bs,ExpertOutDim,1)
        Gate_B_Out = Gate_B_Out.squeeze(2)#维度(bs,ExpertOutDim)  
        Gate_B_Out = torch.reshape(Gate_B_Out, ExpertOutDim)
        
        # if self.GateNum == 3:
        #     return Gate_A_Out,Gate_Shared_Out,Gate_B_Out
        # else:
        return Gate_A_Out,Gate_B_Out

####################### MODELS #######################
class CGC(nn.Module):
    #FeatureDim-输入数据的维数;ExpertOutDim-每个Expert输出的维数;TaskExpertNum-任务特定专家数;CommonExpertNum-共享专家数;n_task-任务数(gate数)
    def __init__(self,in_channels, nepoch,FeatureDim,TaskExpertNum,CommonExpertNum,n_task=2): 
        super(CGC, self).__init__()

        '''一层CGC'''
        self.CGC = Extraction_Network(in_channels, FeatureDim,TaskExpertNum,CommonExpertNum,GateNum=2)
        
        '''TowerA'''
        self.tower1_1 = SingleConv(128, 5, kernel_size=(1,nepoch), stride=1, padding=0)  # sleep stage: conv2d + batchnorm + relu,
        self.tower1_2 = OutSleepStage(5, 5)

        '''TowerB'''
        self.tower2_1 = SingleConv(128, 256, kernel_size=3, stride=1, padding=1)           # main: conv2d + batchnorm + relu
        self.tower2_2 = SingleConv(256, 256, kernel_size=3, stride=1, padding=0)           # main: conv2d + batchnorm + relu
        self.tower2_3 = SingleConv(256, 256, kernel_size=2, stride=1, padding=0)           # main: conv2d + batchnorm + relu
        self.tower2_4 = OutDiagnosis(1024,hidden_channels=256)
        
    def forward(self, x):
          
        Gate_A_Out,Gate_B_Out = self.CGC(x, x, x)
         
        ss = self.tower1_1(Gate_A_Out)
        ss = self.tower1_2(ss)
        ss = torch.squeeze(ss, 3)

        d = self.tower2_1(Gate_B_Out) 
        d = self.tower2_2(d)
        d = self.tower2_3(d)
        d = self.tower2_4(d)
        
        return ss,d

# Model = PLE(FeatureDim=X_train.shape[1],ExpertOutDim=64,TaskExpertNum=1,CommonExpertNum=1).to(device)
# optimizer = torch.optim.Adam(Model.parameters(), lr=0.01)
# loss_func = nn.MSELoss().to(device)

# nParams = sum([p.nelement() for p in Model.parameters()])
# print('* number of parameters: %d' % nParams)