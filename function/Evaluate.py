import numpy as np

from entity.EdgeDevice import *
from function.ToolFunction import *
from function.Heal import *

def evaluate(P, Y, allLegalPaths,serviceList, deviceList, Z):
    SERVICE_NUM = len(Y)
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
    # max_gamma_N = max_network_balancing_loss(serviceList, deviceList, Z)
    fitness = 0
    arr_rate_sum = 0
    for s in deviceList:
        if isinstance(s, AccessPoint):
            for i in s.arrivalRate:
                arr_rate_sum = arr_rate_sum + i
    # 性能计算


    T = 0
    # max_T = 0
    for q in deviceList:
        if (isinstance(q, AccessPoint) == 0):
            break
        for w in serviceList:
            t_w_q = 0
            lamda_w_q = q.arrivalRate[w.serviceId]
            d_w = w.requestSize
            v_q = q.wirelessTransRate
            mu_w = w.executeRate
            path_w_q = getCurrentPath_w_q(P, w.serviceId, q.deviceId)
            while path_w_q == -1:
                P = initP(Y, allLegalPaths, deviceList, serviceList)
                path_w_q = getCurrentPath_w_q(P, w.serviceId, q.deviceId)
            kkk = 0
            for i in range(1, len(path_w_q)):
                kkk = kkk + 1 / Z[path_w_q[i - 1]][path_w_q[i]]
            T = T + lamda_w_q * d_w * (1 / v_q + 1 / mu_w + kkk)
            # 最大时间：最长路径除以最慢速度
            # max_T = max_T + lamda_w_q * d_w * (1/v_q + 1/mu_w + (SERVER_NUM+SWITCH_NUM)/(avg_ZTransRate-bia_ZTransRate))
    # T_norm = T/max_T

    # 资金开销计算
    beta_sum = 0
    for w in serviceList:
        beta_sum = beta_sum + w.unitPrice
    # C_max = len(deviceList)*beta_sum
    C = 0
    for i in range(0, len(Y)):
        for j in range(0, len(Y[0])):
            C = C + serviceList[i].unitPrice * Y[i][j]
    # 归一化
    # C_Y = C_max - C
    # C_Y_norm = C_Y/C_max
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
        for q in deviceList:
            if (isinstance(q, AccessPoint) == 0):
                break
            for w in serviceList:
                path_w_q = getCurrentPath_w_q(P, w.serviceId, q.deviceId)
                if e[0] in path_w_q and P[w.serviceId][q.deviceId][e[1]] == 1:
                    lamda_w_q = q.arrivalRate[w.serviceId]
                    d_w = w.requestSize
                    gamma_e = gamma_e + lamda_w_q * d_w
        gamma.append(gamma_e)
    # 计算方差
    gamma_N = np.var(gamma)
    # print("gamma_N:%f"%(gamma_N))
    # 计算资源平衡损失
    gamma_S = 0
    for w in serviceList:
        sigma_w = np.var(Y[w.serviceId])
        # print("sigma_w:%f" % (sigma_w))
        gamma_S = gamma_S + sigma_w
    # print("gamma_S:%f"%(gamma_S))

    # L_max = max_gamma_N + 0.25 * SERVICE_NUM
    # L_P_Y = (max_gamma_N - gamma_N) + L2 - gamma_S
    # L_P_Y_norm = 1 - gamma_N_norm + (0.25*SERVICE_NUM - gamma_S)/(0.25*SERVICE_NUM)
    T = T/arr_rate_sum
    L_P_Y = b2 * (a2 * (L_1 - gamma_N) + c2) + b3 * (a3 * (L_2 - gamma_S) - c3)
    C_Y = b1 * (a1 * (C_budget - C) + c1)
    rho = T / (C_Y + L_P_Y)
    return rho

def evaluate2(P, Y, allLegalPaths,serviceList, deviceList, Z):
    SERVICE_NUM = len(Y)
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
    # max_gamma_N = max_network_balancing_loss(serviceList, deviceList, Z)
    fitness = 0
    # 性能计算
    T = 0
    # max_T = 0
    for q in deviceList:
        if (isinstance(q , AccessPoint) == 0):
            break
        for w in serviceList:
            t_w_q = 0
            lamda_w_q = q.arrivalRate[w.serviceId]
            d_w = w.requestSize
            v_q = q.wirelessTransRate
            mu_w = w.executeRate
            path_w_q = getCurrentPath_w_q(P, w.serviceId, q.deviceId)
            while path_w_q == -1:
                P = initP(Y, allLegalPaths, deviceList, serviceList)
                path_w_q = getCurrentPath_w_q(P, w.serviceId, q.deviceId)
            kkk = 0
            for i in range(1, len(path_w_q)):
                kkk = kkk + 1/Z[path_w_q[i-1]][path_w_q[i]]
            T = T + lamda_w_q * d_w * (1/v_q + 1/mu_w + kkk)
            # 最大时间：最长路径除以最慢速度
            # max_T = max_T + lamda_w_q * d_w * (1/v_q + 1/mu_w + (SERVER_NUM+SWITCH_NUM)/(avg_ZTransRate-bia_ZTransRate))
    # T_norm = T/max_T

    # 资金开销计算
    beta_sum = 0
    for w in serviceList:
        beta_sum = beta_sum + w.unitPrice
    # C_max = len(deviceList)*beta_sum
    C = 0
    for i in range(0, len(Y)):
        for j in range(0, len(Y[0])):
            C = C + serviceList[i].unitPrice * Y[i][j]
    # 归一化
    # C_Y = C_max - C
    # C_Y_norm = C_Y/C_max
    # 网络平衡损失
    gamma = []
    edge = []
    # 获得所有边
    for i in range(0, len(Z)):
        for j in range(0, len(Z[0])):
            if(Z[i][j] > 0):
                edge.append((i, j, Z[i][j]))
    # 计算每个边的吞吐量
    for e in edge:
        gamma_e = 0
        for q in deviceList:
            if (isinstance(q, AccessPoint) == 0):
                break
            for w in serviceList:
                path_w_q = getCurrentPath_w_q(P, w.serviceId, q.deviceId)
                if e[0] in path_w_q and P[w.serviceId][q.deviceId][e[1]] == 1:
                    lamda_w_q = q.arrivalRate[w.serviceId]
                    d_w = w.requestSize
                    gamma_e = gamma_e + lamda_w_q*d_w
        gamma.append(gamma_e)
    # 计算方差
    gamma_N = np.var(gamma)
    #print("gamma_N:%f"%(gamma_N))
    # 计算资源平衡损失
    gamma_S = 0
    for w in serviceList:
        sigma_w = np.var(Y[w.serviceId])
        # print("sigma_w:%f" % (sigma_w))
        gamma_S = gamma_S + sigma_w
    # print("gamma_S:%f"%(gamma_S))

    # L_max = max_gamma_N + 0.25 * SERVICE_NUM
    # L_P_Y = (max_gamma_N - gamma_N) + L2 - gamma_S
    # L_P_Y_norm = 1 - gamma_N_norm + (0.25*SERVICE_NUM - gamma_S)/(0.25*SERVICE_NUM)
    L_P_Y = b2 * (a2 * (L_1 - gamma_N) + c2) + b3 * (a3 * (L_2 - gamma_S) - c3)
    C_Y = b1 * (a1 * (C_budget - C) + c1)
    rho = T / (C_Y + L_P_Y)
    return rho

def evaluateCTL(P, Y, allLegalPaths,serviceList, deviceList, Z):
    SERVICE_NUM = len(Y)
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
    # max_gamma_N = max_network_balancing_loss(serviceList, deviceList, Z)
    fitness = 0
    arr_rate_sum = 0
    for s in deviceList:
        if isinstance(s, AccessPoint):
            for i in s.arrivalRate:
                arr_rate_sum = arr_rate_sum + i
    # 性能计算
    T = 0
    # max_T = 0
    for q in deviceList:
        if (isinstance(q, AccessPoint) == 0):
            break
        for w in serviceList:
            t_w_q = 0
            lamda_w_q = q.arrivalRate[w.serviceId]
            d_w = w.requestSize
            v_q = q.wirelessTransRate
            mu_w = w.executeRate
            path_w_q = getCurrentPath_w_q(P, w.serviceId, q.deviceId)
            while path_w_q == -1:
                P = initP(Y, allLegalPaths, deviceList, serviceList)
                path_w_q = getCurrentPath_w_q(P, w.serviceId, q.deviceId)
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
    for w in serviceList:
        beta_sum = beta_sum + w.unitPrice
    # C_max = len(deviceList) * beta_sum
    C = 0
    for i in range(0, len(Y)):
        for j in range(0, len(Y[0])):
            C = C + serviceList[i].unitPrice * Y[i][j]
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
        for q in deviceList:
            if (isinstance(q, AccessPoint) == 0):
                break
            for w in serviceList:
                path_w_q = getCurrentPath_w_q(P, w.serviceId, q.deviceId)
                if e[0] in path_w_q and P[w.serviceId][q.deviceId][e[1]] == 1:
                    lamda_w_q = q.arrivalRate[w.serviceId]
                    d_w = w.requestSize
                    gamma_e = gamma_e + lamda_w_q * d_w
        gamma.append(gamma_e)
    # 计算方差
    gamma_N = np.var(gamma)
    # print("gamma_N:%f"%(gamma_N))
    # 计算资源平衡损失
    gamma_S = 0
    for w in serviceList:
        sigma_w = np.var(Y[w.serviceId])
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

def evaluateCTL2(P, Y, allLegalPaths,serviceList, deviceList, Z):
    SERVICE_NUM = len(Y)
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
    # max_gamma_N = max_network_balancing_loss(serviceList, deviceList, Z)
    fitness = 0
    # 性能计算
    T = 0
    # max_T = 0
    for q in deviceList:
        if (isinstance(q, AccessPoint) == 0):
            break
        for w in serviceList:
            t_w_q = 0
            lamda_w_q = q.arrivalRate[w.serviceId]
            d_w = w.requestSize
            v_q = q.wirelessTransRate
            mu_w = w.executeRate
            path_w_q = getCurrentPath_w_q(P, w.serviceId, q.deviceId)
            while path_w_q == -1:
                P = initP(Y, allLegalPaths, deviceList, serviceList)
                path_w_q = getCurrentPath_w_q(P, w.serviceId, q.deviceId)
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
    for w in serviceList:
        beta_sum = beta_sum + w.unitPrice
    # C_max = len(deviceList) * beta_sum
    C = 0
    for i in range(0, len(Y)):
        for j in range(0, len(Y[0])):
            C = C + serviceList[i].unitPrice * Y[i][j]
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
        for q in deviceList:
            if (isinstance(q, AccessPoint) == 0):
                break
            for w in serviceList:
                path_w_q = getCurrentPath_w_q(P, w.serviceId, q.deviceId)
                if e[0] in path_w_q and P[w.serviceId][q.deviceId][e[1]] == 1:
                    lamda_w_q = q.arrivalRate[w.serviceId]
                    d_w = w.requestSize
                    gamma_e = gamma_e + lamda_w_q * d_w
        gamma.append(gamma_e)
    # 计算方差
    gamma_N = np.var(gamma)
    # print("gamma_N:%f"%(gamma_N))
    # 计算资源平衡损失
    gamma_S = 0
    for w in serviceList:
        sigma_w = np.var(Y[w.serviceId])
        # print("sigma_w:%f" % (sigma_w))
        gamma_S = gamma_S + sigma_w
    # print("gamma_S:%f"%(gamma_S))
    # 归一化
    # gamma_N_norm = gamma_N / max_gamma_N
    # L_max = max_gamma_N + 0.25 * SERVICE_NUM
    # L_P_Y = (max_gamma_N - gamma_N) + 0.25 * SERVICE_NUM - gamma_S
    # L_P_Y_norm = 1 - gamma_N_norm + (0.25 * SERVICE_NUM - gamma_S) / (0.25 * SERVICE_NUM)

    L_P_Y = b2 * (a2 * (L_1 - gamma_N) + c2) + b3 * (a3 * (L_2 - gamma_S) - c3)
    C_Y = b1 * (a1 * (C_budget - C) + c1)
    rho = T / (C_Y + L_P_Y)
    return rho, C_Y, L_P_Y, T

def evaluateCTL3(P, Y, allLegalPaths,serviceList, deviceList, Z):
    SERVICE_NUM = len(Y)
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
    # max_gamma_N = max_network_balancing_loss(serviceList, deviceList, Z)
    fitness = 0
    # 性能计算
    T = 0
    # max_T = 0
    for q in deviceList:
        if (isinstance(q, AccessPoint) == 0):
            break
        for w in serviceList:
            t_w_q = 0
            lamda_w_q = q.arrivalRate[w.serviceId]
            d_w = w.requestSize
            v_q = q.wirelessTransRate
            mu_w = w.executeRate
            path_w_q = getCurrentPath_w_q(P, w.serviceId, q.deviceId)
            while path_w_q == -1:
                P = initP(Y, allLegalPaths, deviceList, serviceList)
                path_w_q = getCurrentPath_w_q(P, w.serviceId, q.deviceId)
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
    for w in serviceList:
        beta_sum = beta_sum + w.unitPrice
    # C_max = len(deviceList) * beta_sum
    C = 0
    for i in range(0, len(Y)):
        for j in range(0, len(Y[0])):
            C = C + serviceList[i].unitPrice * Y[i][j]
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
        for q in deviceList:
            if (isinstance(q, AccessPoint) == 0):
                break
            for w in serviceList:
                path_w_q = getCurrentPath_w_q(P, w.serviceId, q.deviceId)
                if e[0] in path_w_q and P[w.serviceId][q.deviceId][e[1]] == 1:
                    lamda_w_q = q.arrivalRate[w.serviceId]
                    d_w = w.requestSize
                    gamma_e = gamma_e + lamda_w_q * d_w
        gamma.append(gamma_e)
    # 计算方差
    gamma_N = np.var(gamma)
    # print("gamma_N:%f"%(gamma_N))
    # 计算资源平衡损失
    gamma_S = 0
    for w in serviceList:
        sigma_w = np.var(Y[w.serviceId])
        # print("sigma_w:%f" % (sigma_w))
        gamma_S = gamma_S + sigma_w
    # print("gamma_S:%f"%(gamma_S))
    # 归一化
    # gamma_N_norm = gamma_N / max_gamma_N
    # L_max = max_gamma_N + 0.25 * SERVICE_NUM
    # L_P_Y = (max_gamma_N - gamma_N) + 0.25 * SERVICE_NUM - gamma_S
    # L_P_Y_norm = 1 - gamma_N_norm + (0.25 * SERVICE_NUM - gamma_S) / (0.25 * SERVICE_NUM)

    L_P_Y = b2 * (a2 * (L_1 - gamma_N) + c2) + b3 * (a3 * (L_2 - gamma_S) - c3)
    C_Y = b1 * (a1 * (C_budget - C) + c1)
    rho = T / (C_Y + L_P_Y)
    return rho, C, gamma_N, gamma_S, T

def evaluatePop(POP, allLegalPaths,serviceList, deviceList, Z):
    sumFit = 0
    best = POP[0]
    best_rho = evaluate(best[0], best[1], allLegalPaths, serviceList, deviceList, Z)
    for p in POP:
        rho = evaluate(p[0], p[1], allLegalPaths, serviceList, deviceList, Z)
        if (rho < best_rho):
            best = p
            best_rho = rho
        sumFit = sumFit + rho
    return sumFit / len(POP), best, best_rho

def evaluatePop2(POP, allLegalPaths,serviceList, deviceList, Z):
    sumFit = 0
    best = POP[0]
    best_rho = evaluate(best[0], best[1], allLegalPaths, serviceList, deviceList, Z)
    for p in POP:
        rho = evaluate(p[0], p[1], allLegalPaths, serviceList, deviceList, Z)
        if(rho < best_rho):
            best = p
            best_rho = rho
        sumFit = sumFit + rho
    return sumFit/len(POP), best, best_rho

def evaluatePopCTL(POP, allLegalPaths,serviceList, deviceList, Z):
    sumFit = 0
    best = POP[0]
    best_rho = evaluate(best[0], best[1], allLegalPaths, serviceList, deviceList, Z)
    Cls = []
    Lls = []
    Tls = []
    for p in POP:
        rho, C, L, T = evaluateCTL(p[0], p[1], allLegalPaths, serviceList, deviceList, Z)
        Cls.append(C)
        Lls.append(L)
        Tls.append(T)
        if (rho < best_rho):
            best = p
            best_rho = rho
        sumFit = sumFit + rho
    return Cls, Lls, Tls

def evaluatePopCTL2(POP, allLegalPaths,serviceList, deviceList, Z):
    sumFit = 0
    best = POP[0]
    best_rho = evaluate(best[0], best[1], allLegalPaths, serviceList, deviceList, Z)
    Cls = []
    gamma_Nls = []
    gamma_Sls = []
    Tls = []
    for p in POP:
        rho, C, gamma_N, gamma_S, T = evaluateCTL(p[0], p[1], allLegalPaths, serviceList, deviceList, Z)
        Cls.append(C)
        gamma_Nls.append(gamma_N)
        gamma_Sls.append(gamma_S)
        Tls.append(T)
        if (rho < best_rho):
            best = p
            best_rho = rho
        sumFit = sumFit + rho
    minC = min(Cls)
    maxC = max(Cls)
    min_gamma_N = min(gamma_Nls)
    max_gamma_N = max(gamma_Nls)
    min_gamma_S = min(gamma_Sls)
    max_gamma_S = max(gamma_Sls)
    minT = min(Tls)
    maxT = max(Tls)
    return [minC,maxC,min_gamma_N,max_gamma_N,min_gamma_S,max_gamma_S,minT,maxT]

# 计算L1（复制的上方代码）
def max_network_balancing_loss(serviceList, deviceList, Z):
    gamma = []
    for i in range(0, len(Z)):
        for j in range(0, len(Z[0])):
            if(Z[i][j] > 0):
                gamma.append(0)
    # 计算每个边的吞吐量
    sum_through_out = 0
    for q in deviceList:
        if (isinstance(q, AccessPoint) == 0):
            break
        for w in serviceList:
            sum_through_out = sum_through_out + q.arrivalRate[w.serviceId] * w.requestSize
    gamma.remove(0)
    gamma.append(sum_through_out)
    # 计算方差
    L1 = np.var(gamma)
    return L1











