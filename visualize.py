import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
import random


def graph(graph):
    nodes_cord = graph.x.tolist()
    
    idx = [i for i in range (len(nodes_cord))]
    pos = dict(zip(idx, nodes_cord))

    g = to_networkx(graph, to_undirected=True)
    nx.draw(g, pos, node_size = 100)
    plt.show()

def random_graphs(num, data):
    chosen = [random.randint(0, len(data)) for _ in range(num)]
    for i in chosen:
        graph(data[i])
