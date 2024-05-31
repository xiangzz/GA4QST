import random
import scipy
from entity.Service import *
from entity.EdgeDevice import Device, AccessPoint, Server, Switch
from entity.Parms import prob_Z , AQ_NUM, SERVER_NUM, SWITCH_NUM
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# 随机生成指定数量服务
def generateServices(num, avg_rqSize, bias_rqSize, avg_exeRate, bias_exeRate, avg_price, bias_price):
    serviceList = []
    for i in range(0, num):
        rqSize = random.uniform(avg_rqSize - bias_rqSize, avg_rqSize + bias_rqSize)
        exeRate = random.uniform(avg_exeRate - bias_exeRate, avg_exeRate + bias_exeRate)
        unitPrice = random.uniform(avg_price - bias_price, avg_price + bias_price)
        service = Service(i, rqSize, exeRate, unitPrice)
        serviceList.append(service)
    return serviceList


# 随机生成指定数量各种设备
def generateDevice(num_ap, num_server, num_switch, avg_lamda, bias_lamda, avg_v, bias_v, avg_capacity, bias_capacity,
                   serviceList):
    deviceList = []
    for i in range(0, num_ap):
        wirelessTransRate = random.uniform(avg_v - bias_v, avg_v + bias_v)
        arrivalRate = []
        for service in serviceList:
            lamda = random.uniform(avg_lamda - bias_lamda, avg_lamda + bias_lamda)
            arrivalRate.append(lamda)
        ap = AccessPoint(i, arrivalRate, wirelessTransRate)
        deviceList.append(ap)

    for i in range(num_ap, num_ap + num_server):
        capacity = random.randint(avg_capacity - bias_capacity, avg_capacity + bias_capacity)
        server = Server(i, capacity)
        deviceList.append(server)

    for i in range(num_ap + num_server, num_ap + num_server + num_switch):
        switch = Switch(i)
        deviceList.append(switch)
    return deviceList


#废弃方法
def generateZ(deviceList, avg_transRate, bias_transRate):
    size = len(deviceList)
    Z = np.zeros((size, size), float)

    # 先生成Server和Switch的拓扑，最后加入AccessPoint
    # 生成第AQ_NUM行
    rd = random.randint(AQ_NUM+1, size - 1)
    Z[AQ_NUM][rd] = 1
    # Z[0][rd] = random.uniform(avg_transRate-bias_transRate, avg_transRate+bias_transRate)
    Z[rd][AQ_NUM] = Z[AQ_NUM][rd]
    for j in range(AQ_NUM+1, size):
        if (j == rd):
            continue
        isConnected = (random.random() < prob_Z)
        if (isConnected == 0):
            Z[AQ_NUM][j] = 0
            Z[j][AQ_NUM] = 0
        else:
            Z[AQ_NUM][j] = 1
            # Z[0][j] = random.uniform(avg_transRate-bias_transRate, avg_transRate+bias_transRate)
            Z[j][AQ_NUM] = Z[AQ_NUM][j]
    # 生成第AQ_NUM+1行到倒数第二行的数据
    for i in range(AQ_NUM+1, size - 1):
        haveLink = False
        for j in range(AQ_NUM, i):
            if (Z[i][j] > 0):
                haveLink = True
                rd1 = -1
        # 如果前面没生成连接
        if (haveLink == False):
            rd1 = random.randint(i + 1, size - 1)
            Z[i][rd1] = 1
            # Z[i][rd1] = random.uniform(avg_transRate - bias_transRate, avg_transRate + bias_transRate)
            Z[rd1][i] = Z[i][rd1]
        for j in range(i, size):
            if (j == i):
                continue
            if (j == rd1):
                continue
            isConnected = (random.random() <= prob_Z)
            if (isConnected == 0):
                Z[i][j] = 0
                Z[j][i] = 0
            else:
                Z[i][j] = 1
                # Z[i][j] = random.uniform(avg_transRate - bias_transRate, avg_transRate + bias_transRate)
                Z[j][i] = Z[i][j]
    # 生成最后一行的数据（为了避免最后一个无连接）
    haveLink1 = False
    for j in range(AQ_NUM, size - 1):
        if (Z[size - 1][j] > 0):
            haveLink1 = True
            break
    if (haveLink1 == False):
        rd2 = random.randint(AQ_NUM, size - 2)
        Z[size - 1][rd2] = 1
        # Z[size - 1][rd2] = random.uniform(avg_transRate - bias_transRate, avg_transRate + bias_transRate)
        Z[rd2][size - 1] = Z[size - 1][rd2]

    # 加入AccessPoint
    for i in range(0, AQ_NUM):
        rd3 = random.randint(AQ_NUM, size-1)
        Z[i][rd3] = 1
        # Z[i][rd3] = random.uniform(avg_transRate - bias_transRate, avg_transRate + bias_transRate)
        Z[rd3][i] = Z[i][rd3]
    return Z

# 生成随机拓扑结构的方法（要求是连通的）
def generateZ2(deviceList, avg_transRate, bias_transRate):
    # 图的生成 先生成Server和Switch的拓扑 再把AccessPoint加进去
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
    G = nx.Graph()
    for i in range(AQ_NUM, AQ_NUM + SERVER_NUM + SWITCH_NUM):
        G.add_node(i)
    for i in range(AQ_NUM, AQ_NUM + SERVER_NUM + SWITCH_NUM):
        for j in range(i+1, AQ_NUM + SERVER_NUM + SWITCH_NUM):
            G.add_edge(i, j, weight=random.uniform(avg_transRate-bias_transRate, avg_transRate+bias_transRate))
    #nx.draw_networkx(G)
    #plt.show()
    T = nx.minimum_spanning_tree(G, weight='weight', algorithm='kruskal')
    #nx.draw_networkx(T)
    #plt.show()
    for i in range(AQ_NUM, AQ_NUM + SERVER_NUM + SWITCH_NUM):
        for j in range(i + 1, AQ_NUM + SERVER_NUM + SWITCH_NUM):
            if(random.random() < prob_Z):
                T.add_edge(i, j, weight=random.uniform(avg_transRate-bias_transRate, avg_transRate+bias_transRate))
    #nx.draw_networkx(T)
    #plt.show()
    # 加入AP
    for i in range(0, AQ_NUM):
        T.add_node(i)
        T.add_edge(i, random.randint(AQ_NUM,
                                     AQ_NUM + SERVER_NUM + SWITCH_NUM - 1),
                   weight=random.uniform(avg_transRate-bias_transRate, avg_transRate+bias_transRate))
    nx.draw_networkx(T)
    plt.show()
    nodeList = []
    #调整输出邻接矩阵的顺序（没有这句话会按node的加入顺序输出邻接矩阵）
    for i in range(0, AQ_NUM+SERVER_NUM+SWITCH_NUM):
        nodeList.append(i)
    Z = nx.to_numpy_array(T, nodeList)    # 带权矩阵（权值为传输速率）
    Z1 = nx.to_numpy_array(T, nodeList, weight = None)  # 01矩阵
    #print(Z)
    #print(Z1)
    return Z

