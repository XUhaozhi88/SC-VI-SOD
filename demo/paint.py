import json

# path = '/workspace/mmdetection/demo/train.json'
path = '/workspace/mmdetection/demo/val.json'

with open(path, 'r') as fp:
    files = json.load(fp)
# print(len(files))
iters = []
min, max, mean = [], [], []
for file in files:
    iter = file['Iteration']
    cls = file['Class']
    box = file['Box']

    var_min = box['Var_Min']
    var_max = box['Var_Max']
    var_mean = box['Var_Mean']
    var_std = box['Var_Std']
    # var_min = cls['Var_Min']
    # var_max = cls['Var_Max']
    # var_mean = cls['Var_Mean']

    iters.append(iter // 200)
    i = 3
    min.append(var_mean[i] - var_std[i])
    max.append(var_mean[i] + var_std[i])
    mean.append(var_mean[i])



import matplotlib.pyplot as plt
from matplotlib import pyplot
# plt.style.use('seaborn-whitegrid')
plt.style.use('seaborn-v0_8')
palette = pyplot.get_cmap('Set1')
font1 = {
    # 'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 32,
}
def draw_line(name_of_alg, color_index, avg, r1, r2):
    color=palette(color_index)
#     avg=np.mean(datas,axis=0)
#     std=np.std(datas,axis=0)
#     r1 = list(map(lambda x: x[0]-x[1], zip(avg, std)))#上方差
#     r2 = list(map(lambda x: x[0]+x[1], zip(avg, std)))#下方差
    plt.plot(iters, avg, color=color, label=name_of_alg, linewidth=3.5)
    plt.fill_between(iters, r1, r2, color=color, alpha=0.2)

draw_line("alg1", 1, mean, min, max)
# draw_line("alg2", 2, alldata2)

plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
# plt.xlabel('Time(s)',fontsize=32)
# plt.ylabel('metric',fontsize=32)
plt.legend(loc='upper left',prop=font1)
# plt.title("instance",fontsize=34)
plt.savefig(path.replace('json', 'jpg'))

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import pyplot
# plt.style.use('seaborn-whitegrid')
# palette = pyplot.get_cmap('Set1')
# font1 = {'family' : 'Times New Roman',
# 'weight' : 'normal',
# 'size'   : 32,
# }

# fig=plt.figure(figsize=(20,10))
# iters=list(range(7))
# #这里随机给了alldata1和alldata2数据用于测试
# alldata1=[]#算法1所有纵坐标数据
# data=np.array([2,4,5,8,11,13,15])#单个数据
# alldata1.append(data)
# data=np.array([2,3,6,12,13,13,15])
# alldata1.append(data)
# data=np.array([2,2,7,9,13,14,16])
# alldata1.append(data)
# alldata1=np.array(alldata1)
# alldata2=[]#算法2所有纵坐标数据
# data=np.array([2,4,5,8,10,10,11])#单个数据
# alldata2.append(data)
# data=np.array([3,3,3,6,7,8,10])
# alldata2.append(data)
# data=np.array([3,3,5,5,6,7,9])
# alldata2.append(data)
# alldata2=np.array(alldata2)

# def draw_line(name_of_alg,color_index,datas):
#     color=palette(color_index)
#     avg=np.mean(datas,axis=0)
#     std=np.std(datas,axis=0)
#     r1 = list(map(lambda x: x[0]-x[1], zip(avg, std)))#上方差
#     r2 = list(map(lambda x: x[0]+x[1], zip(avg, std)))#下方差
#     plt.plot(iters, avg, color=color,label=name_of_alg,linewidth=3.5)
#     plt.fill_between(iters, r1, r2, color=color, alpha=0.2)

# draw_line("alg1",1,alldata1)
# draw_line("alg2",2,alldata2)

# plt.xticks(fontsize=22)
# plt.yticks(fontsize=22)
# plt.xlabel('Time(s)',fontsize=32)
# plt.ylabel('metric',fontsize=32)
# plt.legend(loc='upper left',prop=font1)
# plt.title("instance",fontsize=34)
