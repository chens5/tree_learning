import numpy as np
import networkx as nx
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
import ot
import scipy.spatial.distance as distance
import matplotlib.pyplot as plt
import pydot
from networkx.drawing.nx_pydot import *
from tqdm import trange
import time
import tree_estimators as te

def generate_random_dists(num_dists, n):
    dists = np.random.random_sample((num_dists, n))
    norms = np.linalg.norm(dists, ord=1, axis = 1)
    dists = dists/norms[:, None]
    return dists

def generate_random_points(n, dim=2, low=-10, high=10):
    return (high - low) * np.random.rand(n, dim) + low

def generate_distance_metric(points):
    D = np.zeros((len(points), len(points)))
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist = distance.euclidean(points[i], points[j])
            D[i][j] = dist
            D[j][i] = dist
    return D

def format_distance_matrix(D):
    p_index = 0
    param_list = []
    for i in range(D.shape[0]):
        for j in range(D.shape[0]):
            param_list.append(D[i][j])
            D[i][j] = p_index

            p_index += 1
    return nn.Parameter(torch.tensor(param_list))

# Input: a distance matrix M which references indices in a list of parameters,
def mst_transform(M, output_M, params, output_params):
    root = None
    parents = {}
    subtrees = {}
    node_id = {}
    id_num = 1
    tree = nx.Graph()
    tree.add_nodes_from(np.arange(M.shape[0]))

    connected_component_per_vert = {}
    cc_cur_edge = {}
    leaves = []
    for i in range(M.shape[0]):
        node_id[i] = id_num
        leaves.append(id_num)
        connected_component_per_vert[i] = i
        cc_cur_edge[i] = (None, np.inf)
        id_num += 1
    iteration = 0
    p_index = 0
    while not nx.is_connected(tree):
        for i in range(M.shape[0]):
            for j in range(i+1, M.shape[0]):
                edge = (i, j)
                l_index = M[i][j]
                length = params[l_index]
                c1 = connected_component_per_vert[i] #(e1, e2)
                c2 = connected_component_per_vert[j] #(e3, e4)
                if c1 != c2:
                    if length < cc_cur_edge[c1][1]:
                        cc_cur_edge[c1] = (edge, length)
                    if length < cc_cur_edge[c2][1]:
                        cc_cur_edge[c2] = (edge, length)
        new_cc_cur_edge = {}

        for hrn in cc_cur_edge:
            new_root_node = cc_cur_edge[hrn][0]
            height = cc_cur_edge[hrn][1]
            if not tree.has_node(new_root_node) and new_root_node != None:
                with torch.no_grad():
                    output_params[p_index] = height
                root = new_root_node
                tree.add_node(new_root_node, h=height)
                # Gives index of the node
                node_id[new_root_node] = id_num
                new_cc_cur_edge[new_root_node] = (None, np.inf)

                # left and right subtrees.
                verts0 = []
                verts1 = []

                left_subtree = nx.node_connected_component(tree, connected_component_per_vert[new_root_node[0]])

                right_subtree = nx.node_connected_component(tree, connected_component_per_vert[new_root_node[1]])

                tree.add_edge(connected_component_per_vert[new_root_node[0]], new_root_node)
                tree.add_edge(connected_component_per_vert[new_root_node[1]], new_root_node)
                for v in left_subtree:
                    if v not in parents:
                        parents[v] = new_root_node
                    if type(v) is not tuple:
                        connected_component_per_vert[v] = new_root_node
                        verts0.append(v)
                for v in right_subtree:
                    if v not in parents:
                        parents[v] = new_root_node
                    if type(v) is not tuple:
                        connected_component_per_vert[v] = new_root_node
                        output_M[v, verts0] = p_index
                        verts1.append(v)

                subtrees[new_root_node] = [verts0, verts1]
                for i in verts0:
                    output_M[i, verts1] = p_index
                id_num += 1
                p_index += 1
        cc_cur_edge = new_cc_cur_edge
        iteration += 1

    node_id[root] = 0
    for i in range(M.shape[0]):
        #tree.nodes[i]['h'] = torch.min(M[i])
        output_M[i][i] = p_index
        with torch.no_grad():
            output_params[p_index] = 0.0
        p_index += 1

    p = [0]*tree.number_of_nodes()
    for node in parents:
        parent = parents[node]
        p[node_id[node]] = node_id[parent]
    p[0] = -1
    return tree, parents, p, leaves, root

def format_distributions( dist):
    formatted_dist = []
    for i in range(len(dist)):
        if dist[i] > 0:
            formatted_dist.append((i, dist[i]))
    return formatted_dist

def tree_wasserstein(UM, parameters, solver, mu, rho):
    solver.tree_query(mu, rho)
    matching = solver.return_matching()
    masses = solver.return_mass()
    return torch.sum(torch.tensor(masses) * parameters[UM[matching[0], matching[1]]])

def convert_to_distance_mat(M):
    for i in range(M.shape[0]):
        for j in range(i + 1, M.shape[0]):
            M[i][j] = (2 * M[i][j] - M[i][i] - M[j][j]) * 0.5
