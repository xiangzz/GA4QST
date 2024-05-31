from function.ToolFunction import *
from function.GenerateRandData import *
from function.Initialization import *

# Algorithm 5
def solutionHeal(P, Y, capacity, allLegalPaths, deviceList, serviceList):
    if existOverLoadedServer(Y, capacity) == False:
        return P, Y
    else:
        PVH(Y, capacity)
        try:
            migInfo = SPR(Y, capacity)
            # P = initP(Y, allLegalPaths, deviceList, serviceList)
            NetR(migInfo, P, Y, allLegalPaths, deviceList, serviceList)
        except IndexError:
            P = initP(Y, allLegalPaths, deviceList, serviceList)



# capacity是常量
def PVH(Y, capacity):
    Yo = capacity - Y.sum(0)
    #print(Yo)
    while Yo.sum() <= 0:
        # 将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y
        # 比如argsort([2, -1, -3]),输出[2, 1, 0]
        vIdx = Yo.argsort()
        sIdx = Y.sum(1).argsort()[::-1]
        for i in range(0, len(capacity)):
            v = vIdx[i]
            removed = False
            for j in range(0, len(Y)):
                sw = sIdx[j]
                if Y[sw][v] >= 1:
                    Y[sw][v] = Y[sw][v] - 1
                    removed = True
                    break
            if removed:
                break
        Yo = capacity - Y.sum(0)
    # 添加缺失的服务
    Yt = Y.sum(1)
    for i in range(0, len(Yt)):
        if(Yt[i] <= 0):
            w, h = randMECServerWithDupInstances(Y)
            if h < 0 :
                h = random.randint(0, len(Y[0])-1)
                Y[i][h] = Y[i][h] + 1
            else:
                Y[w][h] = Y[w][h] - 1
                Y[i][h] = Y[i][h] + 1

def SPR(Y, capacity):
    Yo = capacity - Y.sum(0)
    migInfo = []
    # 遍历各服务器
    while existOverLoadedServer(Y, capacity):
        h = randOverloadedMECServer(Y, capacity)
        # 保存过载服务器的服务实例种类
        servicesOfh = set()
        for i in range(0, len(Y)):
            if Y[i][h] > 0:
                servicesOfh.add(i)
        h1 = randMigMECServer(Y, h, capacity)
        servicesOfh1 = set()
        for i in range(0, len(Y)):
            if Y[i][h1] > 0:
                servicesOfh1.add(i)
        # 可以迁移的候选实例集
        servicesDif = servicesOfh.difference(servicesOfh.intersection(servicesOfh1))
        # 随机选出一个
        rd = servicesDif.pop()
        Y[rd][h] = Y[rd][h] - 1
        Y[rd][h1] = Y[rd][h1] + 1
        migInfo.append((rd, serverIdToDeviceId(h), serverIdToDeviceId(h1)))
        Yo = capacity - Y.sum(0)
    return migInfo

# NetR未测试
def NetR(migInfo, P, Y,  allLegalPaths, deviceList, serviceList):
    # inf：元组(w, u, v)
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
    AQ_NUM = ap_num
    SERVER_NUM = server_num
    SWITCH_NUM = switch_num
    for inf in migInfo:
        if(Y[inf[0]][inf[1]-AQ_NUM] > 0):
            continue
        p_v = []
        for i in range(0, AQ_NUM):
            # 获取a_q接收的s_w请求的路由路径
            #print("NetR")
            path = getCurrentPath_w_q(P, inf[0], i)
            # 如果路径的终点是v
            while path == -1:
                P = initP(Y, allLegalPaths, deviceList, serviceList)
                path = getCurrentPath_w_q(P, inf[0], i)
            if path[-1] == inf[1]:
                p_v.append(path)
        for p in p_v:
            # 原路径归零
            initializePath(P, inf[0], p)
            # 随机给一条新路径
        for p in p_v:
            candidatePaths = getLP_i_to_j(p[0], inf[2], allLegalPaths)
            selectedPath = randPath(candidatePaths)
            if isPathLegal(selectedPath) == False:
                return 0
            # print("接入点%d 服务%d所选中路径为" % (ap.deviceId, s.serviceId))
            # print(path)
            for i in range(0, len(selectedPath)):
                if selectedPath[i] == inf[2]:
                    break
                if selectedPath[i]>=AQ_NUM and selectedPath[i]<AQ_NUM+SERVER_NUM and Y[inf[0]][selectedPath[i]-AQ_NUM] > 0:
                    break
                if P[inf[0]][selectedPath[i]].sum() >= 1:
                    break
                else:
                    P[inf[0]][selectedPath[i]][selectedPath[i + 1]] = 1
                    if P[inf[0]][selectedPath[i+1]][selectedPath[i]] == 1:
                        P = initP(Y, allLegalPaths, deviceList, serviceList)
                        return


# 重新生成P矩阵的方法
def initP(Y, allLegalPaths, deviceList, serviceList):
    SERVICE_NUM = len(Y)
    P = np.zeros([SERVICE_NUM, len(deviceList), len(deviceList)])
    for ap in deviceList:
        # 判断是否遍历完ap
        if (isinstance(ap, AccessPoint) == 0):
            break
        for s in serviceList:
            # 先随机选一个目标服务器
        
            destServerId = randDest(Y, s.serviceId)

            destDeviceId = serverIdToDeviceId(destServerId)
            # 随机获得一条从ap到目标的路径
            paths = getLP_i_to_j(ap.deviceId, destDeviceId, allLegalPaths)
            path = randPath(paths)
            if isPathLegal(path) == False:
                return 0

            for i in range(0, len(path)):
                if path[i] == destDeviceId:
                    break
                if P[s.serviceId][path[i]].sum() >= 1:
                    print
                    break
                if isinstance(deviceList[path[i]], Server) and Y[s.serviceId][path[i]-AQ_NUM] > 0:
                    break
                else:
                    P[s.serviceId][path[i]][path[i + 1]] = 1
                    if P[s.serviceId][path[i+1]][path[i]] == 1:
                        print("init")
                        print(P[s.serviceId][path[i]].sum())
    return P






# 经测试可正常使用
# Y = np.array([[20, 2, 1],
#               [0, 0, 0],
#               [3, 10, 1],
#               [1, 15, 1]])
# capacity = np.array([3, 5, 3])
# PVH(Y, capacity)
# print(Y)
# migInfo = SPR(Y, capacity)
# print(Y)
# print(migInfo)


