# -*- coding=utf8 -*-
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import product
import threading
from function.Initialization import legalPathsInit
from function.Heal import *
from function.Evaluate import *

# 服务器数，交换机数，边缘接入点数，服务种类数
N, K, Q, W = 4, 8, 5, 4
V = N + K + Q
PHI_MIN, PHI_MAX = 50, 1000
VQ_MIN, VQ_MAX = 0.5, 10
POP_SIZE = 20
N_GEN = 10
N_Elite = 3
CXP, MTP = 0.8, 0.1
PMTP = 0.5  # 正向变异概率
Y_star = [5, 3, 2, 4]  # 长度取决于N

paths_from_ap_to_server = {}

np.random.seed(3)

def init_network_topo():
    """
        初始化网络拓扑
    :param node_num:
    :param prob:
    :return: Z
    """
    G = nx.Graph()
    # 先加入服务器，交换机
    G.add_nodes_from(range(N + K))
    # 随机化边权重
    count = (N + K) ** 2
    G.add_weighted_edges_from([(i, j, w) for
                               ((i, j), w) in zip(product(range(N + K), range(N + K)),
                                                  np.random.random(count) * (PHI_MAX - PHI_MIN) + PHI_MIN) if i > j])
    # 用最小生成树构建基础连通图
    T = nx.minimum_spanning_tree(G)
    G = None
    # 添加access point进图
    T.add_nodes_from(range(N + K, V))
    # 选择这些ap的连接交换机(这里设置的是ap只连m_k)
    nodes_connected_ap = [i for i in range(N, N + K)]
    np.random.shuffle(nodes_connected_ap)
    nodes_connected_ap = nodes_connected_ap[:Q]
    # 为ap的连接添加边
    for a_q in range(N + K, V):
        T.add_edge(a_q, nodes_connected_ap[a_q - N - K],
                   weight=np.around(np.random.random() * (PHI_MAX - PHI_MIN) + PHI_MIN, decimals=2))
    # 继续为M∪H添加边
    dispatcher = [i for i in range(N + K)]
    while np.random.random() < 0.7:
        np.random.shuffle(dispatcher)
        u, v = dispatcher[: 2]
        T.add_edge(u, v, weight=np.around(np.random.random() * (PHI_MAX - PHI_MIN) + PHI_MIN, decimals=2))
    nodes = [i for i in range(V)]
    Z = nx.to_numpy_array(T, nodes)
    Z = np.around(Z, decimals=2)

    nx.draw_networkx(T, node_color=['g'] * N + ['#FFF000'] * K + ['r'] * Q)
    plt.show()
    return Z


class SIThread(threading.Thread):
    def __init__(self, pop, i):
        threading.Thread.__init__(self)
        self.pop = pop
        self.i = i

    def run(self):
        sol = solution_init()
        self.pop[self.i] = sol


def rand_avail_server_for_service(Y, w):
    h_target = None
    H = [tmp for tmp in range(N)]
    np.random.shuffle(H)
    for h in H:
        # 当且仅当h还有余量，且h上没有部署过w，h才是可用的
        if np.sum(Y, 1)[h] < Y_star[h] and Y[w, h] < 1:
            h_target = h
            break
    return h_target


def solution_init():
    # sol = int(np.random.random() * 100)
    Y = np.zeros((W, N))
    P = np.zeros((W, V, V))
    # 首先，为每个服务至少都部署1次
    for w in range(W):
        h_target = rand_avail_server_for_service(Y, w)
        Y[w, h_target] += 1
    # 其次，为每个服务随机再次部署若干次
    for w in range(W):
        while np.random.random() < 0.5:
            h_target = rand_avail_server_for_service(Y, w)
            if not h_target:
                Y[w, h_target] += 1
    # 最后，生成路径
    for (w, q) in product(range(W), range(N + K, V)):
        v = np.random.choice([h for h in range(N) if Y[w, h] == 1])
        p_wq = np.random.choice(paths_from_ap_to_server[(q, v)])
        for k, u in enumerate(p_wq):
            if u == v:
                break
            _u = p_wq[k + 1]
            if P[w, u, _u] == 1:
                break
            else:
                P[w, u, _u] = 1
    sol = (Y, P)
    return sol


# 初始化网络拓扑
Z = init_network_topo()
# Phi表示传输速率，float
Phi = Z.copy()
# Z表示邻接矩阵，binary
Z = np.int64(Z > 0)
# 调用路径生成函数
all_the_paths_without_ap = legalPathsInit(Z)
for foo in all_the_paths_without_ap:
    if len(foo[0]) > 1:
        for the_path in foo:
            dst = the_path[-1]
            src = the_path[0]
            if (0 <= dst < N) and (np.any(Z[src, N + K: V])):
                for a_q in range(N + K, V):
                    if Z[src, a_q] == 1:
                        src = a_q
                        break
                if not (src, dst) in paths_from_ap_to_server:
                    paths_from_ap_to_server[(src, dst)] = []
                paths_from_ap_to_server[(src, dst)].append([src] + the_path)
# 释放多余空间
all_the_paths_without_ap = None


def evaluateFitness(sol):
    fitness = evaluate(sol[1], sol[0], serviceList, deviceList, Z)
    return fitness

def population_init():
    """
    种群初始化
    :return:
    """
    pop = [None] * POP_SIZE
    sol_init_threads = []
    for i in range(POP_SIZE):
        init_thread = SIThread(pop, i)
        init_thread.start()
        sol_init_threads.append(init_thread)
    for t in sol_init_threads:
        t.join()
    vals = map(evaluate, pop)
    return pop, vals


def select_elite(pop, vals, n_elite):
    elites = []
    vals1 = np.array(list(vals))  # 转格式
    for i in range(n_elite):
        elites.append(pop[np.argsort(vals1)[i]]) 
    return elites


def select_sol(pop, vals):
    f_sol, m_sol = None, None
    vals1 = np.array(list(vals))  # 转格式
    select_probability_sum = (1/vals1/sum(1/vals1)).cumsum()  # 累计概率
    temp = []
    for i in range(2):
        rand = np.random.random()
        for j in range(len(vals1)):
            if j == 0:
                if rand <= select_probability_sum[j]:
                    temp.append(pop[j])
            else:
                if (rand > select_probability_sum[j-1]) and (rand < select_probability_sum[j]):
                    temp.append(pop[j])
    f_sol = temp[0]
    m_sol = temp[1]
    return f_sol, m_sol


def crossover(f_sol, m_sol):
    Y_f = f_sol[0]
    Y_m = m_sol[0]
    # instances to be exchanged
    w = random.randint(1, len(f_sol[0]) - 1)
    # generate new children
    sigma_f_1 = Y_f[:w].copy()
    sigma_f_2 = Y_f[w:].copy()
    sigma_m_1 = Y_m[:w].copy()
    sigma_m_2 = Y_m[w:].copy()
    # child solution standardization
    Y_s = np.vstack((sigma_f_1, sigma_m_2))
    Y_d = np.vstack((sigma_m_1, sigma_f_2))
    s_sol = (Y_s, f_sol[1])
    d_sol = (Y_d, m_sol[1])
    # heal the solutions
    s_sol = (solutionHeal(f_sol[1], s_sol, capacity, allLegalPaths))  # SH(P_f , Y_s, Y ⋆);
    d_sol = (solutionHeal(f_sol[1], d_sol, capacity, allLegalPaths))
    return s_sol, d_sol


def mutate(sol):
    sol_new = None
    # 正向变异
    if np.random.random() < PMTP:
        while True:
            w = int(np.random.random()*W) 
            v = rand_avail_server_for_service(sol[0],w)
            sol[0][w][v]=1 
            sol[1][w][v][v]=1  # 伪代码内容是这样的，但不懂
    # 负向变异
    else:
        while True:
            w = int(np.random.random()*W) 
            v = np.random.choice([h for h in range(N) if sol[0][w, h] == 1]) 
            p = paths_from_ap_to_server[(w, v)]
            initializePath(sol[1],w,p)
            break
    sol_new = sol
    return sol_new


def ga_for_cbst():
    pop, vals = population_init()
    for n_gen in range(N_GEN):
        cur_pop_size = 0
        elites = select_elite(pop, vals, N_Elite)
        cur_pop_size += len(elites)
        new_pop = elites
        while cur_pop_size < POP_SIZE:
            father_sol, mother_sol = select_sol(pop, vals)
            if np.random.random() < CXP:
                son_sol, daughter_sol = crossover(father_sol, mother_sol)
            else:
                son_sol, daughter_sol = father_sol, mother_sol
            if np.random.random() < MTP:
                son_sol = mutate(son_sol)
            if np.random.random() < MTP:
                daughter_sol = mutate(daughter_sol)
            new_pop.append(son_sol)
            new_pop.append(daughter_sol)
        map(evaluate, new_pop)
        pop = new_pop

print('dd')
