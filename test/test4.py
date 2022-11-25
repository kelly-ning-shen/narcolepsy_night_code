import numpy as np

annotations = np.array([0,2,3,1,0])
# erridx = [i for i,x in enumerate(annotations) if x==-1]
erridx = np.where(annotations==-1)[0]
# erridx = [10,11]

if len(erridx)>0 and (erridx[0] == 0 or erridx[-1] == len(annotations)-1):
    erridx_del = erridx # 默认都在头或尾
    if erridx.size != erridx[-1]-erridx[0]+1:
        # 如果中间还有缺失的
        # 只删去开头或结尾的（需要找到连续的idx）
        # 找到[0,1,2,3]
        # 这里不存在头和尾都异常的
        erridx_d = np.diff(erridx)
        idx = np.where(erridx_d==1)[0] # idx: index of erridx_d
        erridx_del = erridx[idx[0]:idx[-1]+2] # 需要删除的erridx
    annotations = np.delete(annotations,erridx_del)
        
print(annotations)

# a = np.array([0,1,2,3,4,5])
# b = np.delete(a,np.arange(3,6))
# print(b)