
import numpy as np
# 从服务序号w开始crossover
import random
from function.Evaluate import *
from function.ToolFunction import *
from function.Heal import *

def crossover(Y_f, Y_m):
    w = random.randint(1, len(Y_f)-1)
    sigma_f_1 = Y_f[:w].copy()
    sigma_f_2 = Y_f[w:].copy()
    sigma_m_1 = Y_m[:w].copy()
    sigma_m_2 = Y_m[w:].copy()
    Y_s = np.vstack((sigma_f_1, sigma_m_2))
    Y_d = np.vstack((sigma_m_1, sigma_f_2))
    return Y_s, Y_d

# 随机交叉两个点之间的基因
def crossover_2point(Y_f, Y_m):
    SERVICE_NUM = len(Y_f)
    SERVER_NUM = len(Y_f[0])
    Chrom1 = Y_f.reshape(1, -1)[0]
    Chrom2 = Y_m.reshape(1, -1)[0]
    n1, n2 = np.random.randint(0, len(Chrom1), 2)
    if n1 > n2:
        n1, n2 = n2, n1
    # crossover at the points n1 to n2
    seg1, seg2 = Chrom1[n1:n2].copy(), Chrom2[n1:n2].copy()
    Chrom1[n1:n2], Chrom2[n1:n2] = seg2, seg1
    Y_s = np.around(Chrom1.reshape(SERVICE_NUM, SERVER_NUM))
    Y_d = np.around(Chrom2.reshape(SERVICE_NUM, SERVER_NUM))
    return Y_s, Y_d

def mutate(P, Y, capacity, deviceList, serviceList, allLegalPaths):
    # positive mutation
    if np.random.random() <= prob_positive_mutate:
        w = random.randint(0, len(serviceList)-1)
        v = randAvailableMECServer(Y, w, capacity)
        Y[w][v] = 1
        for i in range(0, len(P[w][v])):
            P[w][serverIdToDeviceId(v)][i] = 0
    # negative mutation
    else:
        w, v = randMECServerWithDupInstances(Y)
        if w != -1:
            Y[w][v] = 0
            P = initP(Y, allLegalPaths, deviceList, serviceList)


def selectElite(POP, N_ELITE, allLegalPaths, serviceList, deviceList, Z):
    Elites = []
    fits = []
    FitV = []
    fit_back_sum = 0
    for sol in POP:
        t = (sol, evaluate(sol[0], sol[1], allLegalPaths, serviceList, deviceList, Z))
        FitV.append(1/t[1])
        fits.append(t)
        fit_back_sum = fit_back_sum + 1/t[1]
    haveBeenSelected = []
    FitV = FitV - min(FitV) + 1e-10
    sel_prob = FitV / sum(FitV)
    sel_index = np.random.choice(range(len(POP)), size=N_ELITE, replace=False, p=sel_prob)
    for k in sel_index:
        Elites.append(POP[k])
    # for i in range(0, N_ELITE):
    #     rd = random.uniform(0, fit_back_sum)
    #     k = 0
    #     for f in range(0, len(fits)):
    #         if f in haveBeenSelected:
    #             continue
    #         k = k + 1/fits[f][1]
    #         if(k>=rd):
    #             Elites.append(fits[f][0])
    #             haveBeenSelected.append(f)
    #             fit_back_sum = fit_back_sum - 1/fits[f][1]
    #             break
    return Elites


def selectSol(POP, allLegalPaths, serviceList, deviceList, Z):
    parents = selectElite(POP, 2, allLegalPaths, serviceList, deviceList, Z)
    return parents[0], parents[1]