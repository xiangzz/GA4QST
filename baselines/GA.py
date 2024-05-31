import pickle

from function.ToolFunction import *
from function.Heal import *
from function.Evaluate import *
from entity.Parms import *
from entity.EdgeDevice import *
from entity.Service import *
from sko.GA import GA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
'''
min rho(P, Y)
'''

def obj_func(p):
    Y = np.around(p.reshape(SERVICE_NUM, SERVER_NUM))
    # scikit-opt包会对所有p进行计算，即使p是不合约束的。所以对那些不合约束的p取最大值
    try:
        P = initP(Y, allLegalPaths, deviceList, serviceList)
    except ValueError:
        return 1
    return evaluate(P, Y, allLegalPaths, serviceList, deviceList, Z)

file = '../data/Data2022-05-06-11-23-50.pkl'
f = open(file, 'rb')
dataList1 = pickle.load(f)
f.close()
serviceList = dataList1[0]
deviceList = dataList1[1]
capacity = dataList1[2]
Z = dataList1[3]
allLegalPaths = dataList1[4]
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

ga = GA(func=obj_func, n_dim=SERVICE_NUM*SERVER_NUM,
        size_pop=100, max_iter=40,
        lb=lb, ub=ub,
        constraint_eq=constraint_eq, constraint_ueq=constraint_ueq, precision=1e0)

best_x, best_y, n_gen_cost_time = ga.run()
ga_generation_x = ga.generation_best_X
ga_best_rho = []
for i in range(0, len(ga_generation_x)):
    ga_best_rho.append(obj_func(ga_generation_x[i]))
print(ga_best_rho)
print(ga.generation_best_Y)

Y = np.around(best_x.reshape(SERVICE_NUM, SERVER_NUM))
Y_history = pd.DataFrame(ga.all_history_Y)
fig, ax = plt.subplots(2, 1)
ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
Y_history.min(axis=1).cummin().plot(kind='line')
plt.show()
print('best_x:', Y, '\n', 'best_y:', min(ga.generation_best_Y))
print(n_gen_cost_time)
