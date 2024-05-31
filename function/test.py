import random
import scipy
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

G = nx.cycle_graph(4)
G.add_edge(0, 3, weight=2)
nx.draw_networkx(G)
plt.show()
T = nx.minimum_spanning_tree(G)
sorted(T.edges(data=True))
[(0, 1, {}), (1, 2, {}), (2, 3, {})]
nx.draw_networkx(T)
plt.show()