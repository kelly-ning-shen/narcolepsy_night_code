import torch

a = torch.tensor([[[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]],[[[11,12,13],[14,15,16]],[[17,18,19],[110,111,112]]]])
print(a)
print(a.shape)
bs = a.shape[0]
b = torch.reshape(a,(bs,2,-1))
print(b)
print(b.shape)
c = torch.reshape(b,a.shape)
print(c)
print(c.shape)

print('\n')
g = torch.tensor([[2,3],[5,7]]).unsqueeze(2)
print(g.shape)
Gate_Shared_Out = torch.matmul(b.transpose(1,2),g)#维度(bs,ExpertOutDim,1)
Gate_Shared_Out = Gate_Shared_Out.squeeze(2)#维度(bs,ExpertOutDim)
print(Gate_Shared_Out.shape)
d = torch.reshape(Gate_Shared_Out,(2,2,3))
print(d)
print(d.shape)