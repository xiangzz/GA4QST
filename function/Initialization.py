import random

import numpy as np
from function.ToolFunction import *
from entity.Parms import *
from function.Initialization import *

# legalPaths是一个三重列表，用getLP_i_to_j(i, j, allLegalPaths)来获取i到j的所有合法路径
# from function.ToolFunction import randAvailableMECServer


def legalPathsInit(Z):
    allLegalPaths = []
    for i in range(0, len(Z)):
        for j in range(0, len(Z)):
            if(i == j):
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



def chromosomeInit(serviceList, deviceList, Z, allLegalPaths):
    SERVICE_NUM = len(serviceList)
    ap_num = 0
    server_num = 0
    switch_num = 0
    for d in deviceList:
        if isinstance(d, AccessPoint):
            ap_num = ap_num + 1
        if isinstance(d, Server):
            server_num = server_num + 1
        if isinstance(d, Switch):
            switch_num = switch_num + 1
    SERVER_NUM = server_num
    SWITCH_NUM = switch_num
    Y = np.zeros([SERVICE_NUM, SERVER_NUM])
    P = np.zeros([SERVICE_NUM, len(deviceList), len(deviceList)])
    Yx = initYx(deviceList)
    capacity = getCapacity(deviceList)
    # Y生成
    if Yx.sum() < SERVICE_NUM:
        print("服务数量超出可承载数量！")
        return Y
    for s in serviceList:
        k = randAvailableMECServer(Y, s.serviceId, capacity)
        Y[s.serviceId][k] = Y[s.serviceId][k] + 1
        Yx[k] = Yx[k] - 1    # 这一句也可以用upgradeYx 但是效率低
    for s in serviceList:
        while random.random() < prob_y:
            Yo = capacity - Y.sum(0)
            if(Yo.sum() <= 0):
                break
            k = randAvailableMECServer(Y, s.serviceId, capacity)
            if k < 0:
                break
            Y[s.serviceId][k] = Y[s.serviceId][k] + 1
            Yx[k] = Yx[k] - 1  # 这一句也可以用upgradeYx 但是效率低
    # P生成
    for ap in deviceList:
        # 判断是否遍历完ap
        if (isinstance(ap, AccessPoint) == 0):
            break
        for s in serviceList:
            # 先随机选一个目标服务器
            destServerId = randDest(Y, s.serviceId)
            destDeviceId = serverIdToDeviceId(destServerId)
            # print("接入点%d 服务%d所选中目标为" % (ap.deviceId, s.serviceId))
            # print(destDeviceId)
            # 随机获得一条从ap到目标的路径
            paths = getLP_i_to_j(ap.deviceId, destDeviceId, allLegalPaths)
            path = randPath(paths)

            # print("接入点%d 服务%d所选中路径为" % (ap.deviceId, s.serviceId))
            # print(path)
            for i in range(0, len(path)):
                if path[i] == destDeviceId:
                    break
                if P[s.serviceId][path[i]].sum() >= 1:
                    break
                if isinstance(deviceList[path[i]], Server) and Y[s.serviceId][path[i]-AQ_NUM] > 0:
                    break
                else:
                    P[s.serviceId][path[i]][path[i + 1]] = 1
                    if P[s.serviceId][path[i+1]][path[i]] == 1:
                        print("generate")
                        print(P[s.serviceId][path[i]].sum())
    return P, Y


def population_init():
    pops = [None] * N_POP
    sol_init_threads = []



# 其他方法
def initYx(deviceList):
    Yx = []
    for d in deviceList:
        if (isinstance(d, Server)):
            Yx.append(d.capacity)

    return np.array(Yx)

def getCapacity(deviceList):
    capacity = []
    for d in deviceList:
        if (isinstance(d, Server)):
            capacity.append(min(d.capacity, SERVICE_NUM))

    return np.array(capacity)

def randAvailableMECServer(Y, w, capacity):
    Yo = capacity - Y.sum(0)
    availableServerID = []
    for i in range(0, len(Yo)):
        if Yo[i] > 0 and Y[w][i] < 1:
            availableServerID.append(i)
    if len(availableServerID) <= 0:
        return -1
    return availableServerID[random.randint(0, len(availableServerID) - 1)]

def randDest(Y, serviceId):
    candidateServerID = []
    for i in range(0, len(Y[0])):
        if Y[serviceId][i] > 0:
            candidateServerID.append(i)
    if len(candidateServerID) <= 0:
        k = random.randint(0, len(Y[serviceId]) - 1)
        Y[serviceId][k] = 1
        return k
    return candidateServerID[random.randint(0, len(candidateServerID) - 1)]

def serverIdToDeviceId(Id):
    return Id + AQ_NUM

def getLP_i_to_j(i, j, allLegalPaths):
    return allLegalPaths[i*SUM_NUM + j]

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


