import entity.Parms
from function.ToolFunction import *
from function.Heal import *
from function.Evaluate import *
from entity.Parms import *
from entity.EdgeDevice import *
from entity.Service import *
from sko.DE import DE
from sko.GA import GA
from sko.PSO import PSO
from sko.SA import SA
from sko.AFSA import AFSA
import numpy as np
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

file = 'data/Data2022-05-26-20-35-38.pkl'
import os
filename = file[0: 28]
filename = filename.strip()
filename = filename.rstrip("\\")
isExists = os.path.exists(filename)
if not isExists:
    os.makedirs(filename)
else:
    print(filename+"已存在")


str = filename + '/' + 'dataSize' + time_str
csv_name = str + '.csv'
csv_file = open(csv_name, 'w', newline='', encoding='gbk')
f = open(file, 'rb')
dataList1 = pickle.load(f)
f.close()
serviceList = dataList1[0]
deviceList = dataList1[1]
capacity = dataList1[2]
Z = dataList1[3]
allLegalPaths = dataList1[4]

if __name__ == '__main__':
    serviceNum = []
    serviceNum.append(SERVICE_NUM)

    # 5 - 10
    for i in range(0, 5):
        # 参数变化过程(0.1 ~ 20 MB , 4MB间隔)
        for s in serviceList:
            s.requestSize = 4 + 4*i
            identifier = s.requestSize
        # 实验过程
        POP = []
        # 生成种群
        for m in range(0, N_POP):
            # 生成个体基因
            P, Y = chromosomeInit(serviceList, deviceList, Z, allLegalPaths)
            POP.append([P, Y])
        avg_fit, best_in_POP, best_rho = evaluatePop(POP, allLegalPaths, serviceList, deviceList, Z)

        bestPY_in_all_gen = best_in_POP
        rho2, C_bestPY_in_all_gen, L_bestPY_in_all_gen, T_bestPY_in_all_gen \
            = evaluateCTL(bestPY_in_all_gen[0], bestPY_in_all_gen[1], allLegalPaths, serviceList, deviceList, Z )

        # print(best_in_POP[1])
        print("初始平均Rho为%f，最优解Rho为%f" % (avg_fit, best_rho))
        # plot值
        generation = []
        g_best_rho = [identifier]
        Cls = []
        Lls = []
        Tls = []

        min_max_table = []

        best_rho_in_all_generation = best_rho
        for n_gen in range(0, N_GEN):
            elites = selectElite(POP, N_ELITE, allLegalPaths, serviceList, deviceList, Z)
            new_POP = elites
            while len(new_POP) < N_POP:
                father_sol, mother_sol = selectSol(POP, allLegalPaths, serviceList, deviceList, Z)
                if np.random.random() < prob_crossover:
                    Y_son, Y_daughter = crossover(father_sol[1], mother_sol[1])
                    PVH(Y_son, capacity)
                    PVH(Y_daughter, capacity)
                    P_son = initP(Y_son, allLegalPaths, deviceList, serviceList)
                    P_daughter = initP(Y_daughter, allLegalPaths, deviceList, serviceList)
                    solutionHeal(P_son, Y_son, capacity, allLegalPaths, deviceList, serviceList)
                    solutionHeal(P_daughter, Y_daughter, capacity, allLegalPaths, deviceList, serviceList)
                else:
                    Y_son = father_sol[1].copy()
                    Y_daughter = mother_sol[1].copy()
                    P_son = father_sol[0].copy()
                    P_daughter = mother_sol[0].copy()
                if np.random.random() < prob_mutate:
                    mutate(P_son, Y_son, capacity, deviceList, serviceList, allLegalPaths)
                if np.random.random() < prob_mutate:
                    mutate(P_son, Y_son, capacity, deviceList, serviceList, allLegalPaths)
                new_POP.append([P_son, Y_son])
                new_POP.append([P_daughter, Y_daughter])
            POP = new_POP
            avg_fit, best_in_POP, best_rho = evaluatePop(POP, allLegalPaths, serviceList, deviceList, Z)
            rho1, C, L, T = evaluateCTL(best_in_POP[0], best_in_POP[1], allLegalPaths, serviceList, deviceList, Z)
            # 5.26 新增
            min_max_row = evaluatePopCTL2(POP, allLegalPaths, serviceList, deviceList, Z)
            min_max_table.append(min_max_row)
            print(min_max_row)



            if best_rho < best_rho_in_all_generation:
                Cls.append(C)
                Lls.append(L)
                Tls.append(T)
                C_bestPY_in_all_gen, L_bestPY_in_all_gen, T_bestPY_in_all_gen = C, L, T
                bestPY_in_all_gen = best_in_POP
                best_rho_in_all_generation = best_rho
            else:
                Cls.append(C_bestPY_in_all_gen)
                Lls.append(L_bestPY_in_all_gen)
                Tls.append(T_bestPY_in_all_gen)

            g_best_rho.append(min(best_rho, best_rho_in_all_generation))
            generation.append(n_gen)
            print("第%d轮，平均Rho为%f，最优解Rho为%f" % (n_gen, avg_fit, best_rho))
            # print("best Y")
            # print(best_in_POP[1])
        np.array(g_best_rho)
        Cls.insert(0, 'C')
        Lls.insert(0, 'L')
        Tls.insert(0, 'T')
        # 保存数据
        writer = csv.writer(csv_file)
        writer.writerow(g_best_rho)
        writer.writerow(Cls)
        writer.writerow(Lls)
        writer.writerow(Tls)
        writer.writerow([])

    print(min_max_table)
    csv_file.close()