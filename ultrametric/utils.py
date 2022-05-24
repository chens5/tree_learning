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
    for i in trange(len(points)):
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

def mst_transform(input_mat, output_mat, input_params, output_params):
    M = input_params[input_mat].detach().numpy()
    span_tree = minimum_spanning_tree(M)
    edges = []
    for i in range(len(span_tree.nonzero()[0])):
        edges.append([(span_tree.nonzero()[0][i],span_tree.nonzero()[1][i]), span_tree.data[i]])
    edges.sort(key=lambda x: x[1])
    parents = [None] * (len(edges) + M.shape[0])
    subtrees = [None] * (len(edges) + M.shape[0])
    # initialize leaves and parents
    leaves = []
    leaf_component = []
    parent_index = 1
    parameter_index = 0
    for i in range(M.shape[0]):
        leaves.append(parent_index)
        leaf_component.append(parent_index)
        subtrees[parent_index] = [parent_index]
        with torch.no_grad():
            output_mat[i][i] = parameter_index
            output_params[parameter_index] = 0.0
            # output_params[parameter_index] = torch.min(M[i])
            parameter_index += 1
        parent_index += 1
        
    leaf_component = np.array(leaf_component)
    for value in edges:
        edge = value[0]
        # get leaf node index in parents list
        leaf_index1 = leaves[edge[0]]
        leaf_index2 = leaves[edge[1]]
        left_subtree = None
        right_subtree = None
        if parent_index == len(parents):
            parent_index = 0
        if parents[leaf_index1] == None:
            parents[leaf_index1] = parent_index
            left_subtree = [leaf_index1 - 1]
        else:
            # index of current highest parent
            current_highest_parent = leaf_component[edge[0]]
            parents[current_highest_parent] = parent_index
            left_subtree = subtrees[current_highest_parent]
            
        if parents[leaf_index2] == None:
            parents[leaf_index2] = parent_index
            right_subtree = [leaf_index2 - 1]
        else:
            current_highest_parent = leaf_component[edge[1]]
            parents[current_highest_parent] = parent_index
            right_subtree = subtrees[current_highest_parent]
        subtrees[parent_index] = left_subtree + right_subtree
        with torch.no_grad():
            output_mat[left_subtree, :][:, right_subtree] = parameter_index
            output_mat[right_subtree, :][:, left_subtree] = parameter_index
            output_params[parameter_index] = value[1]
            parameter_index += 1
        leaf_component[left_subtree] = parent_index
        leaf_component[right_subtree] = parent_index
        
        parent_index += 1
    parents[0] = -1
    return parents, leaves

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
