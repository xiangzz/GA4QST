import matplotlib.pyplot as plt

from entity.Service import *
from entity.Parms import *
from entity.EdgeDevice import *
from function.GenerateRandData import *
from function.Initialization import *
from function.ToolFunction import *
from function.Heal import *
from function.Evaluate import *
from function.EvolutionaryStrategies import *

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # 生成服务列表 常量
    serviceList = generateServices(SERVICE_NUM, avg_rqSize, bias_rqSize, avg_exeRate, bias_exeRate, avg_price, bias_price)
    # 生成设备列表 常量
    deviceList = generateDevice(AQ_NUM, SERVER_NUM, SWITCH_NUM, avg_lamda, bias_lamda, avg_v, bias_v, avg_c, bias_c,
                                serviceList)
    capacity = getCapacity(deviceList)
    # 生成拓扑结构 常量
    Z = generateZ2(deviceList, avg_ZTransRate, bia_ZTransRate)
    # 根据Z找到所有合法路径 也是个常量
    allLegalPaths = legalPathsInit(Z)
    # 计算L1（吞吐量最大方差 所有吞吐量集中在一条连接）
    L1 = max_network_balancing_loss(serviceList, deviceList, Z)
    print(L1)
    #
    P, Y = chromosomeInit(serviceList, deviceList, Z, allLegalPaths)
    POP = []

    # 生成种群
    for i in range(0, N_POP):
        # 生成个体基因
        P, Y = chromosomeInit(serviceList, deviceList, Z, allLegalPaths)
        POP.append([P, Y])
    avg_fit, best_in_POP, best_rho = evaluatePop(POP, allLegalPaths,serviceList, deviceList, Z)
    #print(best_in_POP[1])
    print("初始平均Rho为%f，最优解Rho为%f" % (avg_fit, best_rho))
    # plot值
    generation = []
    g_best_rho = []
    best_rho_in_all_generation = best_rho
    for n_gen in range(0, N_GEN):
        elites = selectElite(POP, N_ELITE, allLegalPaths,serviceList, deviceList, Z)
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
        avg_fit, best_in_POP, best_rho = evaluatePop(POP, allLegalPaths,serviceList, deviceList, Z)
        g_best_rho.append(min(best_rho, best_rho_in_all_generation))
        generation.append(n_gen)
        print("第%d轮，平均Rho为%f，最优解Rho为%f" % (n_gen, avg_fit, best_rho))
        # print("best Y")
        # print(best_in_POP[1])
    plt.plot(generation, g_best_rho)
    plt.show()












