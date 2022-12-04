import math

preds = [[
    [-0.3618,0.2934],
    [1.8247,0.8872],
    [-0.2192,-0.5751],
    [2.0236,-0.5462],
    [0.8538, -0.0333]
],[
    [-0.1776,0.4205],
    [2.0221, 0.9147],
    [-1.0514,-0.8845],
    [1.1581,-0.8651],
    [0.0178, -0.2894]
],[
    [-0.7603,-0.7837],
    [-0.2788,-0.1452],
    [-0.8556,0.2014],
    [2.4627,-2.5102],
    [0.0562,-0.3370]
]]

preds_exp = [[
    [0,0],
    [0,0],
    [0,0],
    [0,0],
    [0,0]
],[
    [0,0],
    [0,0],
    [0,0],
    [0,0],
    [0,0]
],[
    [0,0],
    [0,0],
    [0,0],
    [0,0],
    [0,0]
]]

labels = [[1,1],[2,0],[2,2]]

output = 1.8955
logitsums = []
batchsize = len(preds)
for b in range(batchsize):
    C = len(preds[b]) # num of class
    tmp = [0,0]
    for c in range(C):
        D = len(preds[b][c]) # another dimension
        for d in range(D):
            preds_exp[b][c][d] = math.exp(preds[b][c][d])
            tmp[d] += preds_exp[b][c][d]
    logitsums.append(tmp)

loss = 0
for b in range(batchsize):
    for d in range(D):
        idx = labels[b][d]
        loss0 = -preds[b][idx][d] + math.log(logitsums[b][d])
        loss += loss0

crossentropyloss = loss/(batchsize*D)
print(crossentropyloss) # 1.8954865311243303
# 在每个对应位置上做交叉熵，然后取均值！对的！