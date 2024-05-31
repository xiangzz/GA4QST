import pickle
import datetime
import matplotlib.pyplot as plt
import csv
import networkx as nx
import numpy as np
from itertools import product


file = 'data/Data2022-05-26-20-35-38.pkl'
f = open(file, 'rb')
dataList1 = pickle.load(f)
f.close()
serviceList = dataList1[0]
deviceList = dataList1[1]
capacity = dataList1[2]
Z = dataList1[3]
allLegalPaths = dataList1[4]

v_num, _ = Z.shape
graph = nx.Graph()

graph.add_nodes_from([i for i in range(v_num)])
graph.add_edges_from([(i, j) for (i, j) in product(range(v_num), range(v_num)) if Z[i, j] > 0])
nx.draw_networkx(graph)
plt.show()