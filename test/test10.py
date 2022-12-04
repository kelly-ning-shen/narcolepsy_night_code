# cross entropy loss
# This criterion computes the cross entropy loss between input logits and target.
import torch
import torch.nn as nn

# Example of target with class indices
loss = nn.CrossEntropyLoss()
BATCH_SIZE = 10
preds =  torch.randn(BATCH_SIZE, 5, 30, requires_grad=True)
labels = torch.empty(BATCH_SIZE,30, dtype=torch.long).random_(5) # 这里不用有类别数C的维度（不需要将label转化为one-hot），体现在取值上
output = loss(preds, labels)
output.backward()

print('hello world')