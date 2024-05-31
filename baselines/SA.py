from function.ToolFunction import *
from function.Heal import *
from function.Evaluate import *
from entity.Parms import *
from entity.EdgeDevice import *
from entity.Service import *
from sko.SA import SA
import numpy as np
'''
min rho(P, Y)
'''

def obj_func(p):
    Y = np.around(p.reshape(SERVICE_NUM, SERVER_NUM))
    try:
        P = initP(Y, allLegalPaths, deviceList, serviceList)
    except ValueError:
        return 1
    return evaluate(P, Y, allLegalPaths, serviceList, deviceList, Z)


serviceList = generateServices(SERVICE_NUM, avg_rqSize, bias_rqSize, avg_exeRate, bias_exeRate, avg_price, bias_price)
# 生成设备列表 常量
deviceList = generateDevice(AQ_NUM, SERVER_NUM, SWITCH_NUM, avg_lamda, bias_lamda, avg_v, bias_v, avg_c, bias_c,
                                serviceList)
capacity = getCapacity(deviceList)
# 生成拓扑结构 常量
Z = generateZ2(deviceList, avg_ZTransRate, bia_ZTransRate)
# 根据Z找到所有合法路径 也是个常量
allLegalPaths = legalPathsInit(Z)
# 计算L1（吞吐量最大方差 所有吞吐量集中在一条连接）
L1 = max_network_balancing_loss(serviceList, deviceList, Z)

# P, Y = chromosomeInit(serviceList, deviceList, Z, allLegalPaths)
# def obj_func(p):
#     x1, x2, x3 = p
#     return x1 ** 2 + x2 ** 2 + x3 ** 2

def constraint1(p):
    Y = np.around(p.reshape(SERVICE_NUM, SERVER_NUM))
    for j in range(0, len(Y[0])):
        if capacity[j] - Y[:, j].sum() < 0:
            return -1
    for i in range(0, len(Y)):
        if Y[i].sum() < 1:
            return -1
    return 1

constraint_eq = [
]
constraint_ueq = [
    constraint1
]

lb = []
ub = []
for j in range(0, SERVICE_NUM*SERVER_NUM):
    lb.append(0)
    ub.append(1)
P, Y = chromosomeInit(serviceList, deviceList, Z, allLegalPaths)
x0 = []
for sublist in Y :
    for item in sublist:
          x0.append(item)

sa = SA(func=obj_func, x0=x0, T_max=1, T_min=1e-9, q=0.99, L=150, max_stay_counter=10,
                 lb=lb, ub=ub)
sa.run()
sa_best_rho = sa.generation_best_Y[:10]
print(sa_best_rho)