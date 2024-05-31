import random
import scipy
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# 服务器数、接入点数、服务种类数、交换机数
SERVER_NUM = 4
AP_NUM = 3
SERVICE_NUM = 5
SWITCH_NUM = 3
DEVICE_NUM = AP_NUM + SERVER_NUM + SWITCH_NUM

# 服务请求数据大小
rq_size_min = 5
rq_size_max = 15
rq_size_set = []

# 服务单位资源处理速度
exe_rate_min = 5
exe_rate_max = 15
exe_rate_set = []

# 服务单位资源分配成本
price_min = 5
price_max = 15
price_set = []

# 接入点请求率
ap_rate_min = 5
ap_rate_max = 15
ap_rate_set = []

# 接入点无线传输速率
v_min = 5
v_max = 15
v_set = []

# 服务器资源
res_min = 5
res_max = 8
res_set = []

# 路径生成参数
prob_Z = 0.3  # 感觉还可以
z_rate_min = 5
z_rate_max = 15

# DE参数
N_POP = 100
N_GEN = 20
N_ELITE = 44

# 特殊
RU = 0.3
L_1 = 270000
L_2 = 2.5

# 其他
C_budget = 430
b1 = 0.5
b2 = 0.3
b3 = 0.2
a1 = 1
a2 = 0.00158203125
a3 = 168.75

c1 = 0
c2 = -10.8203125
c3 = -11.875

def generateZ3():
    G = nx.Graph()
    for i in range(AP_NUM, AP_NUM + SERVER_NUM + SWITCH_NUM):
        G.add_node(i)
    for i in range(AP_NUM, AP_NUM + SERVER_NUM + SWITCH_NUM):
        for j in range(i + 1, AP_NUM + SERVER_NUM + SWITCH_NUM):
            G.add_edge(i, j, weight=random.uniform(z_rate_min, z_rate_max))
    # nx.draw_networkx(G)
    # plt.show()
    T = nx.minimum_spanning_tree(G, weight='weight', algorithm='kruskal')
    # nx.draw_networkx(T)
    # plt.show()
    for i in range(AP_NUM, AP_NUM + SERVER_NUM + SWITCH_NUM):
        for j in range(i + 1, AP_NUM + SERVER_NUM + SWITCH_NUM):
            if random.random() < prob_Z:
                T.add_edge(i, j, weight=random.uniform(z_rate_min, z_rate_max))
    # nx.draw_networkx(T)
    # plt.show()
    # 加入AP
    for i in range(0, AP_NUM):
        T.add_node(i)
        T.add_edge(i, random.randint(AP_NUM,
                                     AP_NUM + SERVER_NUM + SWITCH_NUM - 1),
                   weight=random.uniform(z_rate_min, z_rate_max))
    nx.draw_networkx(T)
    plt.show()
    nodeList = []
    # 调整输出邻接矩阵的顺序（没有这句话会按node的加入顺序输出邻接矩阵）
    for i in range(0, AP_NUM + SERVER_NUM + SWITCH_NUM):
        nodeList.append(i)
    Z = nx.to_numpy_array(T, nodeList)  # 带权矩阵（权值为传输速率）
    # Z1 = nx.to_numpy_array(T, nodeList, weight=None)  # 01矩阵
    # print(Z)
    # print(Z1)
    return Z


def legalPathsInit(Z):
    allLegalPaths = []
    for i in range(0, len(Z)):
        for j in range(0, len(Z)):
            if (i == j):
                allLegalPaths.append([[i]])
            else:
                ijPaths = []
                generateLegalPathsFromU_V(i, j, Z, [], ijPaths)
                allLegalPaths.append(ijPaths)
    return allLegalPaths


def generateLegalPathsFromU_V(uid, vid, Z, currentPath, paths):
    if (len(currentPath) == 0):
        currentPath.append(uid)
    cPathCopy = currentPath[:]
    if (uid == vid):
        # cPathCopy.append(vid)
        paths.append(cPathCopy)
    else:
        for pid in range(0, len(Z)):
            # 这个条件寻找所有和u有连接的点，去掉之前经过的点，发起递归找路
            if (Z[uid][pid] > 0 and (pid in cPathCopy) == False):
                cPathCopy = currentPath[:]
                cPathCopy.append(pid)
                generateLegalPathsFromU_V(pid, vid, Z, cPathCopy, paths)


def system_init():
    # 请求大小、单位资源实例处理速率、单位资源实例价格
    for i in range(0, SERVICE_NUM):
        rq_size_set.append(random.uniform(rq_size_min, rq_size_max))
        exe_rate_set.append(random.uniform(exe_rate_min, exe_rate_max))
        price_set.append(random.uniform(price_min, price_max))
    # 接入点请求率
    for i in range(0, AP_NUM):
        ap_rate_set_1 = []
        for j in range(0, SERVICE_NUM):
            ap_rate_set_1.append(random.uniform(ap_rate_min, ap_rate_max))
        ap_rate_set.append(ap_rate_set_1)
    # 接入点无线传输速率
    for i in range(0, AP_NUM):
        v_set.append(random.uniform(v_min, v_max))
    for i in range(0, SERVER_NUM):
        res_set.append(random.uniform(res_min, res_max))

from sko.DE import DE
from sko.GA import GA
from sko.PSO import PSO
from sko.SA import SA
from sko.AFSA import AFSA
import pickle
import datetime

def overloaded(Y):
    Rx = res_set - Y.sum(0)
    for i in Rx:
        if i < 0:
            return True
    return False

def randDest(Y, serviceId):
    candidateServerID = []
    for i in range(0, len(Y[0])):
        if Y[serviceId][i] > RU:
            candidateServerID.append(i)
    return candidateServerID[random.randint(0, len(candidateServerID) - 1)]

def randPath(paths):
    sum = 0
    for path in paths:
        sum = sum + SERVER_NUM+SWITCH_NUM-len(path)+1
    rd = random.randint(0, sum)
    sum = 0
    for path in paths:
        sum = sum + SERVER_NUM+SWITCH_NUM-len(path)+1
        if(rd <= sum):
            return path

def isPathLegal(path):
    for p in path:
        if path.count(p) >= 2:
            return False

def initP(Y):
    P = np.zeros([SERVICE_NUM, DEVICE_NUM, DEVICE_NUM])
    for i in range(0, AP_NUM):
        for j in range(0, SERVICE_NUM):
            destServerId = randDest(Y, j)
            destDeviceId = destServerId + AP_NUM
            paths = allLegalPaths[i*DEVICE_NUM + destDeviceId]
            path = randPath(paths)
            if isPathLegal(path) == False:
                return 0
            for k in range(0, len(path)):
                node = path[k]
                if path[k] == destDeviceId:
                    break
                if P[j][path[k]].sum() >= 1:
                    print
                    break
                if AP_NUM <= path[k] < AP_NUM+SERVER_NUM and Y[j][path[k] - AP_NUM] > RU:
                    break
                else:
                    P[j][path[k]][path[k + 1]] = 1
                    if P[j][path[k + 1]][path[k]] == 1:
                        print("init")
                        print(P[j][path[k]].sum())
    return P

def getCurrentPath_w_q(P, w, q):
    i = q
    path = [q]
    while P[w][i].sum() > 0:
        for j in range(0, len(P[w][i])):
            if P[w][i][j] > 0:
                if j in path:
                    print("出现loop")
                    #print(P[w])
                    print(path, j)
                    return -1
                path.append(j)
                break
        i = j
    return path

def evaluateCTL(P, Y, allLegalPaths, Z):

    fitness = 0
    arr_rate_sum = 0
    for s in range(0, AP_NUM):
        for i in ap_rate_set[s]:
            arr_rate_sum = arr_rate_sum + i
    # 性能计算
    T = 0
    # max_T = 0
    for q in range(0, AP_NUM):
        for w in range(0, SERVICE_NUM):
            t_w_q = 0
            lamda_w_q = ap_rate_set[q][w]
            d_w = rq_size_set[w]
            v_q = v_set[q]
            mu_w = exe_rate_set[w]
            path_w_q = getCurrentPath_w_q(P, w, q)
            while path_w_q == -1:
                P = initP(Y)
                path_w_q = getCurrentPath_w_q(P, w, q)
            kkk = 0
            for i in range(1, len(path_w_q)):
                kkk = kkk + 1 / Z[path_w_q[i - 1]][path_w_q[i]]
            T = T + lamda_w_q * d_w * (1 / v_q + 1 / mu_w + kkk)
            # 最大时间：最长路径除以最慢速度
            # max_T = max_T + lamda_w_q * d_w * (
            #            1 / v_q + 1 / mu_w + (SERVER_NUM + SWITCH_NUM) / (avg_ZTransRate - bia_ZTransRate))
    # T_norm = T / max_T

    # 资金开销计算
    beta_sum = 0
    for w in range(0, SERVICE_NUM):
        beta_sum = beta_sum + price_set[w]
    # C_max = len(deviceList) * beta_sum
    C = 0
    for i in range(0, len(Y)):
        for j in range(0, len(Y[0])):
            C = C + price_set[i] * Y[i][j]
    # 归一化
    # C_Y = C_max - C
    # C_Y_norm = C_Y / C_max
    # 网络平衡损失
    gamma = []
    edge = []
    # 获得所有边
    for i in range(0, len(Z)):
        for j in range(0, len(Z[0])):
            if (Z[i][j] > 0):
                edge.append((i, j, Z[i][j]))
    # 计算每个边的吞吐量
    for e in edge:
        gamma_e = 0
        for q in range(0, AP_NUM):
            for w in range(0, SERVICE_NUM):
                path_w_q = getCurrentPath_w_q(P, w, q)
                if e[0] in path_w_q and P[w][q][e[1]] == 1:
                    lamda_w_q = ap_rate_set[q][w]
                    d_w = rq_size_set[w]
                    gamma_e = gamma_e + lamda_w_q * d_w
        gamma.append(gamma_e)
    # 计算方差
    gamma_N = np.var(gamma)
    # print("gamma_N:%f"%(gamma_N))
    # 计算资源平衡损失
    gamma_S = 0
    for w in range(0, SERVICE_NUM):
        sigma_w = np.var(Y[w])
        # print("sigma_w:%f" % (sigma_w))
        gamma_S = gamma_S + sigma_w
    # print("gamma_S:%f"%(gamma_S))
    # 归一化
    # gamma_N_norm = gamma_N / max_gamma_N
    # L_max = max_gamma_N + 0.25 * SERVICE_NUM
    # L_P_Y = (max_gamma_N - gamma_N) + 0.25 * SERVICE_NUM - gamma_S
    # L_P_Y_norm = 1 - gamma_N_norm + (0.25 * SERVICE_NUM - gamma_S) / (0.25 * SERVICE_NUM)
    T = T / arr_rate_sum
    L_P_Y = b2 * (a2 * (L_1 - gamma_N) + c2) + b3 * (a3 * (L_2 - gamma_S) - c3)
    C_Y = b1 * (a1 * (C_budget - C) + c1)
    rho = T / (C_Y + L_P_Y)
    return rho, C_Y, L_P_Y, T

best_rho = 1
best_PY_rho = []

def obj_func(p):
    Y = p.reshape(SERVICE_NUM, SERVER_NUM)
    if overloaded(Y):
        rho_dict[1] = [-1, -1, -1]
        return 20
    try:
        P = initP(Y)
    except ValueError:
        return 20
    rho, C_Y, L_P_Y, T = evaluateCTL(P, Y, allLegalPaths, Z)
    rho_dict[rho] = [C_Y, L_P_Y, T]
    if rho < best_rho:
        best_PY_rho.append([Y, P, rho])
    return rho


def constraint1(p):
    Y = p.reshape(SERVICE_NUM, SERVER_NUM)
    for j in range(0, len(Y[0])):
        if res_set[j] - Y[:, j].sum() < 0:
            return -1
    for i in range(0, len(Y)):
        count = sum(num > RU for num in Y[i])
        if count < 1:
            return -1
    return 0

def getBestRhoUntilNGenList(ls):
    best = ls[0]
    ans = []
    ans.append(ls[0])
    for i in range(1, len(ls)):
        ans.append(min(best, ls[i]))
        if ls[i] < best:
            best = ls[i]
    return ans

constraint_eq = [
]
constraint_ueq = [
    constraint1
]

lb = []
ub = []
for j in range(0, SERVICE_NUM * SERVER_NUM):
    lb.append(0)
    ub.append(2)

Z = generateZ3()
allLegalPaths = legalPathsInit(Z)
system_init()


currtime = datetime.datetime.now()
time_str = datetime.datetime.strftime(currtime, '%Y-%m-%d-%H-%M-%S')

dataList = []
dataList.append(rq_size_set)
dataList.append(exe_rate_set)
dataList.append(price_set)
dataList.append(ap_rate_set)
dataList.append(v_set)
dataList.append(res_set)
dataList.append(Z)
dataList.append(allLegalPaths)
str = 'datade/'+ 'DataDE' + time_str + '.pkl'
f = open(str, 'wb+')
pickle.dump(dataList, f)
f.close()
# 执行DE操作
print("执行DE方法")
rho_dict = {'rho': []}
de = DE(func=obj_func, n_dim=SERVICE_NUM * SERVER_NUM,
        size_pop=N_POP, max_iter=N_GEN,
        lb=lb, ub=ub,
        constraint_eq=constraint_eq, constraint_ueq=constraint_ueq)

best_x, best_y, de_time_cost = de.run()
de_best_rho = de.generation_best_Y
de_best_rho = getBestRhoUntilNGenList(de_best_rho)

C = []
L = []
T = []

print(de_best_rho)

print("执行GA方法")
ga = GA(func=obj_func, n_dim=SERVICE_NUM * SERVER_NUM,
        size_pop=N_POP, max_iter=N_GEN,
        lb=lb, ub=ub,
        constraint_eq=constraint_eq, constraint_ueq=constraint_ueq, precision=1e0)

best_x, best_y, ga_time_cost = ga.run()


ga_best_rho = ga.generation_best_Y
ga_best_rho = getBestRhoUntilNGenList(ga_best_rho)
C = []
L = []
T = []
for r in ga_best_rho:
    C.append(rho_dict[r][0])
    L.append(rho_dict[r][1])
    T.append(rho_dict[r][2])


print(ga_best_rho)


print("执行PSO方法")
pso = PSO(func=obj_func, n_dim=SERVICE_NUM * SERVER_NUM,
          pop=N_POP, max_iter=N_GEN,
          lb=lb, ub=ub,
          constraint_eq=constraint_eq, constraint_ueq=constraint_ueq)
best_x, best_y, pso_time_cost = pso.run()
pso_best_rho = pso.gbest_y_hist
pso_best_rho = np.array(pso_best_rho)
pso_best_rho = pso_best_rho.reshape(1, -1)

C = []
L = []
T = []
pso_best_rho[0] = getBestRhoUntilNGenList(pso_best_rho[0])
for r in pso_best_rho[0]:
    C.append(rho_dict[r][0])
    L.append(rho_dict[r][1])
    T.append(rho_dict[r][2])


print(pso_best_rho)

# SA
print("执行SA方法")
x0 = []
for sublist in best_PY_rho[0][0]:
    for item in sublist:
        x0.append(item)

sa = SA(func=obj_func, x0=x0, T_max=1, T_min=1e-9, q=0.99, L=150, max_stay_counter=N_GEN,
        lb=lb, ub=ub)
x_sa, y_sa, sa_time_cost = sa.run()
sa_best_rho1 = sa.generation_best_Y[:N_GEN]
sa_best_rho = sa_best_rho1[0]
C = []
L = []
T = []
sa_best_rho1 = getBestRhoUntilNGenList(sa_best_rho1)
for r in sa_best_rho1:
    C.append(rho_dict[r][0])
    L.append(rho_dict[r][1])
    T.append(rho_dict[r][2])
print(sa_best_rho1)
# writer.writerow(sa_best_rho1)
# writer.writerow(C)
# writer.writerow(L)
# writer.writerow(T)
# writer.writerow(sa_time_cost[:N_GEN])
# writer.writerow([])

# writer.writerow(getBestRhoUntilNGenList(g_best_rho))
# writer.writerow(getBestRhoUntilNGenList(ga_best_rho))
# writer.writerow(getBestRhoUntilNGenList(de_best_rho))
# writer.writerow(getBestRhoUntilNGenList(pso_best_rho[0]))
# writer.writerow(getBestRhoUntilNGenList(sa_best_rho1))

