import pickle
import datetime
import matplotlib.pyplot as plt
import csv

from entity.Service import *
from entity.Parms import *
from entity.EdgeDevice import *
from function.GenerateRandData import *
from function.Initialization import *
from function.ToolFunction import *
from function.Heal import *
from function.Evaluate import *
from function.EvolutionaryStrategies import *

currtime = datetime.datetime.now()
time_str = datetime.datetime.strftime(currtime, '%Y-%m-%d-%H-%M-%S')
# serviceList = generateServices(SERVICE_NUM, avg_rqSize, bias_rqSize, avg_exeRate, bias_exeRate, avg_price, bias_price)
# # 生成设备列表 常量
# deviceList = generateDevice(AQ_NUM, SERVER_NUM, SWITCH_NUM, avg_lamda, bias_lamda, avg_v, bias_v, avg_c, bias_c,
#                                 serviceList)
# capacity = getCapacity(deviceList)
# # 生成拓扑结构 常量
# Z = generateZ2(deviceList, avg_ZTransRate, bia_ZTransRate)
# # 根据Z找到所有合法路径 也是个常量
# allLegalPaths = legalPathsInit(Z)
# print(capacity)
# # 保存实验数据
# dataList = []
# dataList.append(serviceList)
# dataList.append(deviceList)
# dataList.append(capacity)
# dataList.append(Z)
# dataList.append(allLegalPaths)
# str = 'data/'+ 'Data' + time_str + '.pkl'
# f = open(str, 'wb+')
# pickle.dump(dataList, f)
# f.close()
# 读取数据
# 读取文件名
# 初始超参数


file = 'data/Data2022-05-26-20-35-38.pkl'
import os
filename = file[0: 28]
filename = filename.strip()
filename = filename.rstrip("\\")
isExists = os.path.exists(filename)
if not isExists:
    os.makedirs(filename)
f = open(file, 'rb')
dataList1 = pickle.load(f)
f.close()
serviceList = dataList1[0]
deviceList = dataList1[1]
capacity = dataList1[2]
Z = dataList1[3]
allLegalPaths = dataList1[4]
capacity = getCapacity(deviceList)
print(capacity)
rho_dict = {'rho': []}
arr_rate_sum = 0
for s in deviceList:
    if isinstance(s, AccessPoint):
        for i in s.arrivalRate:
            arr_rate_sum = arr_rate_sum + i

print(arr_rate_sum)