import torch
import numpy as np
from torch_geometric.data import Data
from selected_points_old import bi_graph, bi_graph_norm
import os

_ , Edges = bi_graph_norm()
V , D_Edges = bi_graph()
DATA_SRC = "./points_data/"


def load_dataset(src):
    graphs = []
    for folder in range(2):
        for filename in os.listdir(src + str(folder)):
            if filename[0] == '.':
                continue
            arr = np.load(src + str(folder) + '/' + filename)  # array of points
            arr = norm_points(arr)  # normalize the points between 0 and 1 relative to the y-axis
            cur_graph = build_graph(arr, folder)
            graphs.append(cur_graph)
    return graphs


def build_graph(points, label):
    # node features is defined as (x,y) points
    nodes = torch.tensor(np.asarray([points[i] for i in V]), dtype=torch.float32)
    # edge_index
    edges = torch.tensor(Edges)
    # edge features is distance between the points.
    edges_atrr = torch.tensor([[distance(points[edge[0]], points[edge[1]])] for edge in D_Edges], dtype=torch.float32)
    # global
    y = torch.tensor(label)

    return Data(nodes, edges.t().contiguous(), edges_atrr, y)


def data():
    return load_dataset(DATA_SRC)


def distance(p1, p2):
    dist = np.asarray(p1) - np.asarray(p2)
    return np.sqrt(np.dot(dist, dist))


def norm_points(arr):
    min_y = np.min(arr, axis=0)[0]
    max_y = np.max(arr, axis=0)[1]
    arr = (arr - min_y) * (1 / (max_y - min_y))
    arr = arr - (arr[4] - [0.5, 0.5])
    return arr