import random
from entity.EdgeDevice import *
import numpy as np
from entity.Parms import *
from function.Initialization import *


# 初始化Y*的方法
def initYx(deviceList):
    Yx = []
    for d in deviceList:
        if (isinstance(d, Server)):
            Yx.append(d.capacity)

    return np.array(Yx)

def getCapacity(deviceList):
    capacity = []
    SERVICE_NUM = len(deviceList[0].arrivalRate)
    for d in deviceList:
        if (isinstance(d, Server)):
            capacity.append(min(d.capacity, SERVICE_NUM))

    return np.array(capacity)


# 更新Y*的方法
# Y是一个二维矩阵 Yx是一维向量
def getYx(Y, deviceList):
    Yx = initYx(deviceList)
    Yx = Yx - Y.sum(0)  # sum of each column
    return Yx


# 随机返回一个可用服务器的 服务器 编号(可用：实例小于上限 且无w实例)
def randAvailableMECServer(Y, w, capacity):
    Yo = capacity - Y.sum(0)
    availableServerID = []
    for i in range(0, len(Yo)):
        if Yo[i] > 0 and Y[w][i] < 1:
            availableServerID.append(i)
    if len(availableServerID) <= 0:
        return -1
    return availableServerID[random.randint(0, len(availableServerID) - 1)]

# 随机返回一个实例未达到上限的服务器的 服务器 编号
def randMigMECServer(Y, h, capacity):
    Yo = capacity - Y.sum(0)
    availableServerID = []
    for i in range(0, len(Yo)):
        if Yo[i] > 0:
            for j in range(0, len(Y)):
                if Y[j][i] == 0 and Y[j][h] == 1:
                    availableServerID.append(i)
    if len(availableServerID) > 0:
        return availableServerID[random.randint(0, len(availableServerID) - 1)]
    else:
        return availableServerID[0]



# 随机返回一个有服务s实例的目标服务器的 服务器 编号
def randDest(Y, serviceId):
    candidateServerID = []
    for i in range(0, len(Y[0])):
        if Y[serviceId][i] > 0:
            candidateServerID.append(i)
    return candidateServerID[random.randint(0, len(candidateServerID) - 1)]


# 服务器编号转设备编号(建立于先生成接入点再生成服务器的基础之上)
def serverIdToDeviceId(Id):
    return Id + AQ_NUM


# 生成合法路径的算法 其中currentPath是为了不走回头路
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


# 把Z带权矩阵转换成01矩阵的方法
def Zto01Matrix(Z):
    #print(Z)
    Z1 = np.zeros(np.shape(Z))
    for i in range(0, len(Z)):
        for j in range(0, len(Z)):
            if Z[i][j] > 0:
                Z1[i][j] = 1
    return Z1

# 获取从i到j的所有合法路径
def getLP_i_to_j(i, j, allLegalPaths):
    return allLegalPaths[i*SUM_NUM + j]

# 返回随机路径（路径长度越短选中概率越大）
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

# 随机返回 有多个某服务实例的服务器的设备编号 及 该服务编号
def randMECServerWithDupInstances(Y):
    cadidateServiceId = []
    for i in range(0, len(Y)):
        if Y[i].sum()>1:
            cadidateServiceId.append(i)
    if len(cadidateServiceId) <=0:
        return -1, -1
    rdw = cadidateServiceId[random.randint(0, len(cadidateServiceId)-1)]
    cadidateServerId = []
    for j in range(0, len(Y[rdw])):
        if Y[rdw][j] >= 1:
            cadidateServerId.append(j)
    rdh = cadidateServerId[random.randint(0, len(cadidateServerId)-1)]
    return rdw, rdh
    # w_h_List = []
    # for w in range(0, len(Y)):
    #     for h in range(0, len(Y[0])):
    #         if(Y[w][h] > 1):
    #             w_h_List.append((w, h))
    # return w_h_List[random.randint(0, len(w_h_List)-1)]

# 判断是否存在过载服务器
def existOverLoadedServer(Y, capacity):
    Yo = capacity - Y.sum(0)
    for i in range(0, len(Yo)):
        if Yo[i] < 0:
            return True
    return False

# 随机返回已过载的服务器的 服务器 编号
def randOverloadedMECServer(Y, capacity):
    Yo = capacity - Y.sum(0)
    overLoadedServerID = []
    for i in range(0, len(Yo)):
        if Yo[i] < 0:
            overLoadedServerID.append(i)
    return overLoadedServerID[random.randint(0, len(overLoadedServerID)-1)]

# 获得服务w、接入点q的当前选中路径
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

# 将P矩阵中的对应路径归零
def initializePath(P, w, path):
    for i in range(0, len(path)-1):
        P[w][path[i]][path[i+1]] = 0

        #print(P[w][path[i]][path[i+1]])

def isPLegal(P):
    for i in range(0, len(P)):
        for j in range(0, len(P[0])):
            for k in range(0, len(P[0][0])):
                if P[i][j][k] == 1 and P[i][k][j] == 1:
                    return False

def isPathLegal(path):
    for p in path:
        if path.count(p) >= 2:
            return False

def getBestRhoUntilNGenList(ls):
    best = ls[0]
    ans = []
    ans.append(ls[0])
    for i in range(1, len(ls)):
        ans.append(min(best, ls[i]))
        if ls[i] < best:
            best = ls[i]
    return ans

def overloaded(Y, deviceList):
    capacity = getCapacity(deviceList)
    Yx = capacity - Y.sum(0)
    for i in Yx:
        if i < 0:
            return False
    return True
