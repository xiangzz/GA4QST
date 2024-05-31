import networkx as nx
import numpy as np
from itertools import product
import matplotlib.pyplot as plt

adj_mat = np.array([[0,  0,  0,  0,  0,  0,  0,  1,  0],
 [0,  0,  0,  0,  0,  0,  1,  0,  0],
 [0,  0,  0,  0,  0,  1,  0,  0,  0],
 [0,  0,  0,  0,  1,  0,  0,  0,  0],
 [0,  0,  0,  1,  0,  0,  0,  0,  0],
 [0,  0,  1,  0,  0,  0,  1,  0,  0],
 [0,  1,  0,  0,  0,  1,  0,  0,  0],
 [1,  0,  0,  0,  0,  0,  0,  0,  1],
 [0,  0,  0,  0,  0,  0,  0,  1,  0]]


)


v_num, _ = adj_mat.shape
graph = nx.Graph()

graph.add_nodes_from([i for i in range(v_num)])
graph.add_edges_from([(i, j) for (i, j) in product(range(v_num), range(v_num)) if adj_mat[i, j] == 1])
nx.draw_networkx(graph)
plt.show()