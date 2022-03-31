import numpy as np
import networkx as nx
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
import ot
import scipy.spatial.distance as distance

# Input: a matrix M
# Return: networkx tree and ultramatrix, UM
def matrix_to_um(M, param_matrix):
    root = None
    parents = {}
    subtrees = {}
    tree = nx.Graph()
    tree.add_nodes_from(np.arange(M.shape[0]))
    #UM = np.zeros((M.shape[0], M.shape[0]))
    connected_component_per_vert = {}
    cc_cur_edge = {}
    for i in range(M.shape[0]):
        connected_component_per_vert[i] = i
        cc_cur_edge[i] = (None, np.inf)
    iteration = 0
    max_iteration = 100
    while not nx.is_connected(tree) and iteration < max_iteration:
        for i in range(M.shape[0]):
            for j in range(i+1, M.shape[0]):
                edge = (i, j)
                length = M[i][j]
                c1 = connected_component_per_vert[i] #(e1, e2)
                c2 = connected_component_per_vert[j] #(e3, e4)
                if c1 != c2:
                    if length < cc_cur_edge[c1][1]:
                        cc_cur_edge[c1] = (edge, length)
                    if length < cc_cur_edge[c2][1]:
                        cc_cur_edge[c2] = (edge, length)
        new_cc_cur_edge = {}
        #print(cc_cur_edge)
        #e()
        for hrn in cc_cur_edge:
            new_root_node = cc_cur_edge[hrn][0]
            height = cc_cur_edge[hrn][1]
            print("New root node", new_root_node)
            if not tree.has_node(new_root_node):
                root = new_root_node
                tree.add_node(new_root_node, h=height)
                new_cc_cur_edge[new_root_node] = (None, np.inf)
                verts0 = []
                verts1 = []
                left_subtree = nx.node_connected_component(tree, connected_component_per_vert[new_root_node[0]])
                
                right_subtree = nx.node_connected_component(tree, connected_component_per_vert[new_root_node[1]])
                #print("Right subtree", right_subtree)
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
                        with torch.no_grad():
                            param_matrix[v, verts0] = height
                        verts1.append(v)
                # print("Verts0", verts0)
                # print("Verts1", verts1)
                subtrees[new_root_node] = [verts0, verts1]
                for i in verts0:
                    with torch.no_grad():
                        param_matrix[i, verts1] = height
                    #print(UM)
        print(list(tree.edges))
        h()        
                # for v in nx.node_connected_component(tree, new_root_node):
                #     if type(v) is not tuple:
                #         connected_component_per_vert[v] = new_root_node
        # print(UM)
        # h()

        cc_cur_edge = new_cc_cur_edge
        iteration += 1
    if iteration == max_iteration:
        return 0
    print(list(tree.edges))
    for i in range(M.shape[0]):
        tree.nodes[i]['h'] = M[i][i]
        with torch.no_grad():
            param_matrix[i][i] = M[i][i]
    #     # Assume for now that the h for every vertex
    #     UM[i][i] = M[i][i]
    #     for j in range(i + 1, M.shape[0]):
    #         lca = nx.lowest_common_ancestor(tree, i, j)
    #         UM[i][j] = lca['h']
    #         UM[j][i] = lca['h']
    return tree, parents, subtrees, root

# distributions in numpy
def compute_tree_distance(UM, parents, subtrees, root, mu, rho):
    result = 0
    for i in range(M.shape[0]):
        height = M[i][i]
        parent = parents[i]
        ls = subtrees[parent][0]
        rs = subtrees[parent][1]
        height_parent = (np.sum(UM[ls, :][:, rs]) + np.sum(UM[rs, :][:, ls]))/(len(rs)*len(ls) + len(ls)*len(rs))
        result += abs(mu[i] - rho[i]) * abs(height_parent - height) * 0.5
    for v in subtrees:
        if v != root:
            ls = subtrees[v][0]
            rs = subtrees[v][1]
            mu_mass = np.sum(mu[ls]) + np.sum(mu[rs])
            nu_mass = np.sum(nu[ls]) + np.sum(nu[rs])
            total_mass_moved = abs(mu_mass - nu_mass)
            height = (np.sum(UM[ls, :][:, rs]) + np.sum(UM[rs, :][:, ls]))/(2 * len(rs)*len(ls))
            parent = parents[v]
            ls_parent = subtrees[parent][0]
            rs_parent = subtrees[parent][1]
            height_parent = (np.sum(UM[ls_parent, :][:, rs_parent]) + np.sum(UM[ls_parent, :][:, rs_parent]))/(2*len(rs_parent)*len(ls_parent))
            result += 0.5 * abs(height - height_parent) * total_mass_moved

    return result

def loss_fn(UM, parents, subtrees, D, dists1, dists2):
    loss = 0
    for i in range(len(dists1)):
        mu = dists1[i]
        rho = dists2[i]
        loss += (ot.emd2(D, mu, rho) - compute_tree_distace(UM, parents, subtrees, root, mu, rho))**2
    return loss

# Careful when setting parameters for torch gradient descent
def train_wasserstein(D, dists1, dists2, max_iterations=5):
    M = D
    UM = torch.zeros(M.shape[0], M.shape[0])
    parameters = nn.Parameter(UM)
    optimizer = torch.optim.SGD([UM], lr=0.0001)
    print("finished initialization")
    tree, parents, subtree, root = matrix_to_um(M, UM)
    print("first step done")
    for i in range(max_iterations):
        loss = loss_fn(UM, parents, subtrees, D, dists1, dists2)
        loss.backward()
        optimizer.step()
        tree, parents, subtree, root = matrix_to_um(M, UM)

    return tree, parents, subtree, root, UM


def generate_random_dists(num_dists, n):
    dists = np.random.random_sample((num_dists, n))
    norms = np.linalg.norm(dists, ord=1, axis = 1)
    dists = dists/norms[:, None]
    return dists

def generate_random_points(n, dim=2, low=-10):
    return (high - low) * np.random.rand(n, dim, seed=0) + low

def generate_distance_metric(points):
    D = np.zeros((len(points), len(points)))
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist = distance.euclidean(points[i], points[j])
            D[i][j] = dist
            D[j][i] = dist
    return D

def test_tree():
    M = np.array([[0, 10, 36, 1], [1, 0, 1, 24], [3, 5, 0, 12], [0, 0, 0, 0]])
    M = np.array([[0, 1, 3], [0, 0, 5], [0, 0, 0]])
    start = time.time()
    tree, UM, parents, subtrees, root = matrix_to_um(M)
    end = time.time()
    print("Time for 4 nodes:", end - start)
    print(UM)
    print(list(tree.edges))
    print(parents)
    print(subtrees)
    mu = np.array([2/3, 1/3, 0])
    nu = np.array([2/3, 0, 1/3])
    print(compute_tree_distance(UM, parents, subtrees, root, mu, nu))

def main():
    pointset = generate_random_points(10)
    #D = distance.pdist(pointset, metric='euclidean')
    #print(len(D))
    D = generate_distance_metric(pointset)
    d1 = generate_random_dists(3, 10)
    d2 = generate_random_dists(3, 10)
    train_wasserstein(D, d1, d2)


if __name__ == '__main__':
    main()
