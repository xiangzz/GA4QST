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
# serviceList = generateServices(SERVICE_NUM, avg_rqSize, bias_rqSize, avg_exeRate, bias_exeRate, avg_price, bias_price)
# # 生成设备列表 常量
# deviceList = generateDevice(AQ_NUM, SERVER_NUM, SWITCH_NUM, avg_lamda, bias_lamda, avg_v, bias_v, avg_c, bias_c,
#                                 serviceList)
# capacity = getCapacity(deviceList)
# # 生成拓扑结构 常量
# Z = generateZ2(deviceList, avg_ZTransRate, bia_ZTransRate)
# # 根据Z找到所有合法路径 也是个常量
# allLegalPaths = legalPathsInit(Z)
# print(capacity)
# # 保存实验数据
# dataList = []
# dataList.append(serviceList)
# dataList.append(deviceList)
# dataList.append(capacity)
# dataList.append(Z)
# dataList.append(allLegalPaths)
# str = 'data/'+ 'Data' + time_str + '.pkl'
# f = open(str, 'wb+')
# pickle.dump(dataList, f)
# f.close()
# 读取数据
# 读取文件名
file = 'data/Data2022-05-26-20-35-38.pkl'
import os
filename = file[0: 28]
filename = filename.strip()
filename = filename.rstrip("\\")
isExists = os.path.exists(filename)
if not isExists:
    os.makedirs(filename)
f = open(file, 'rb')
dataList1 = pickle.load(f)
f.close()
serviceList = dataList1[0]
deviceList = dataList1[1]
capacity = dataList1[2]
Z = dataList1[3]
allLegalPaths = dataList1[4]
print(capacity)
rho_dict = {'rho': []}
# baseline 参数
def obj_func(p):
    Y = np.around(p.reshape(SERVICE_NUM, SERVER_NUM))
    try:
        P = initP(Y, allLegalPaths, deviceList, serviceList)
    except ValueError:
        return 1
    rho, C_Y, L_P_Y, T = evaluateCTL(P, Y, allLegalPaths, serviceList, deviceList, Z)
    rho_dict[rho] = [C_Y, L_P_Y, T]
    return rho

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

if __name__ == '__main__':
    cera_s = []
    ga_s = []
    de_s = []
    pso_s = []
    sa_s = []
    for k in range(0, 22):

        POP = []
        C = []
        L = []
        T = []
        # 生成种群
        for i in range(0, N_POP):
            # 生成个体基因
            P, Y = chromosomeInit(serviceList, deviceList, Z, allLegalPaths)
            POP.append([P, Y])
        avg_fit, best_in_POP, best_rho = evaluatePop(POP, allLegalPaths, serviceList, deviceList, Z)
        # print(best_in_POP[1])
        rho1, C_y, L_py, T_py = evaluateCTL(best_in_POP[0], best_in_POP[1], allLegalPaths, serviceList, deviceList, Z)
        bestPY_in_all_gen = best_in_POP
        rho2, C_bestPY_in_all_gen, L_bestPY_in_all_gen, T_bestPY_in_all_gen \
            = evaluateCTL(bestPY_in_all_gen[0], bestPY_in_all_gen[1], allLegalPaths, serviceList, deviceList, Z)

        print("初始平均Rho为%f，最优解Rho为%f" % (avg_fit, best_rho))
        # plot值
        generation = []
        g_best_rho = []
        cost_time_in_n_gen = []
        best_rho_in_all_generation = best_rho
        for n_gen in range(0, N_GEN):
            elites = selectElite(POP, N_ELITE, allLegalPaths, serviceList, deviceList, Z)
            new_POP = elites
            heal_time_cost = 0
            while len(new_POP) < N_POP:
                father_sol, mother_sol = selectSol(POP, allLegalPaths, serviceList, deviceList, Z)
                if np.random.random() < prob_crossover:
                    Y_son, Y_daughter = crossover(father_sol[1], mother_sol[1])
                    heal_start_time = datetime.datetime.now()
                    PVH(Y_son, capacity)
                    PVH(Y_daughter, capacity)
                    P_son = initP(Y_son, allLegalPaths, deviceList, serviceList)
                    P_daughter = initP(Y_daughter, allLegalPaths, deviceList, serviceList)
                    solutionHeal(P_son, Y_son, capacity, allLegalPaths, deviceList, serviceList)
                    solutionHeal(P_daughter, Y_daughter, capacity, allLegalPaths, deviceList, serviceList)
                    heal_finish_time = datetime.datetime.now()
                    heal_time_cost = heal_time_cost + (heal_finish_time - heal_start_time).total_seconds()
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
            evaluate_start_time = datetime.datetime.now()
            avg_fit, best_in_POP, best_rho = evaluatePop(POP, allLegalPaths, serviceList, deviceList, Z)
            evaluate_finish_time = datetime.datetime.now()
            print('evaluate Time:{time_cost}s'
                  .format(time_cost=(evaluate_finish_time - evaluate_start_time).total_seconds()))
            cost_time_in_n_gen.append(heal_time_cost + (evaluate_finish_time - evaluate_start_time).total_seconds())
            rho1, C_y, L_py, T_py = evaluateCTL(best_in_POP[0], best_in_POP[1], allLegalPaths, serviceList, deviceList,
                                                Z)
            if best_rho < best_rho_in_all_generation:
                g_best_rho.append(best_rho)
                C.append(C_y)
                L.append(L_py)
                T.append(T_py)
                C_bestPY_in_all_gen, L_bestPY_in_all_gen, T_bestPY_in_all_gen = C_y, L_py, T_py
                bestPY_in_all_gen = best_in_POP
                best_rho_in_all_generation = best_rho
            else:
                g_best_rho.append(best_rho_in_all_generation)
                C.append(C_bestPY_in_all_gen)
                L.append(L_bestPY_in_all_gen)
                T.append(T_bestPY_in_all_gen)
            generation.append(n_gen)
            print("第%d轮，平均Rho为%f，最优解Rho为%f" % (n_gen, avg_fit, best_rho))
            # print("best Y")
            # print(best_in_POP[1])
        np.array(g_best_rho)

        # 保存数据
        csv_name = filename + '/Scatter' + time_str + '.csv'
        csv_file = open(csv_name, 'w', newline='', encoding='gbk')


        # csv_file.close()

        # plt.plot(generation, g_best_rho)
        # plt.show()
        # GA
        print("执行GA方法")
        ga = GA(func=obj_func, n_dim=SERVICE_NUM * SERVER_NUM,
                size_pop=N_POP, max_iter=N_GEN,
                lb=lb, ub=ub,
                constraint_eq=constraint_eq, constraint_ueq=constraint_ueq, precision=1e0)

        best_x, best_y, ga_time_cost = ga.run()

        # Y = np.around(best_x.reshape(SERVICE_NUM, SERVER_NUM))
        ga_best_rho = ga.generation_best_Y
        C = []
        L = []
        T = []
        for r in ga_best_rho:
            C.append(rho_dict[r][0])
            L.append(rho_dict[r][1])
            T.append(rho_dict[r][2])


        print(ga_best_rho)
        # DE
        print("执行DE方法")
        de = DE(func=obj_func, n_dim=SERVICE_NUM * SERVER_NUM,
                size_pop=N_POP, max_iter=N_GEN,
                lb=lb, ub=ub,
                F=1,
                constraint_eq=constraint_eq, constraint_ueq=constraint_ueq)

        best_x, best_y, de_time_cost = de.run()
        de_best_rho = de.generation_best_Y

        C = []
        L = []
        T = []
        for r in de_best_rho:
            C.append(rho_dict[r][0])
            L.append(rho_dict[r][1])
            T.append(rho_dict[r][2])

        print(de_best_rho)
        # PSO
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
        for r in pso_best_rho[0]:
            C.append(rho_dict[r][0])
            L.append(rho_dict[r][1])
            T.append(rho_dict[r][2])


        print(pso_best_rho)
        # SA
        print("执行SA方法")
        x0 = []
        for sublist in Y:
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
        for r in sa_best_rho1:
            C.append(rho_dict[r][0])
            L.append(rho_dict[r][1])
            T.append(rho_dict[r][2])

        cera_s.append(g_best_rho[-1])
        ga_s.append(ga_best_rho[-1])
        de_s.append(de_best_rho[-1])
        pso_s.append(pso_best_rho[0][-1])
        sa_s.append(sa_best_rho1[-1])

    csv_name = filename + '/Scatter' + time_str + '.csv'
    csv_file = open(csv_name, 'w', newline='', encoding='gbk')
    writer = csv.writer(csv_file)
    writer.writerow(cera_s)
    writer.writerow(ga_s)
    writer.writerow(de_s)
    writer.writerow(pso_s)
    writer.writerow(sa_s)
    csv_file.close()







