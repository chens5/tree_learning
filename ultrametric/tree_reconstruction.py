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
from tqdm import trange, tqdm
import time
import tree_estimators as te
import ot_estimators as ote
from utils import *
import multiprocessing as mp
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from ot_tree import *
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score
import sys
from scipy.stats import mode
import pickle
import argparse
from treeOT import *


def get_distance_matrix(clustertree, n):
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            a = np.zeros(n)
            b = np.zeros(n)
            a[i] = 1
            b[j] = 1
            dist[i][j] = clustertree.pairwiseTWD(a, b)
    return dist

def tree_reconstruction_task(D,D_gds,train_distributions, device='cuda:3', num_dists=10, train_weights=False, lr=0.01, batchsz=64):    

    tree = GPUOptimizedTree(D, device = device)
    tree.to(device)
    #train_distributions = generate_random_dists(num_dists, n)
    
    num_dists = len(train_distributions)
    x = []
    y = []
    d = []
    for i in range(num_dists):
        for j in range(i + 1, num_dists):
            a = train_distributions[i]
            b = train_distributions[j]
            otdistance=ot.emd2(a, b, D_gds)
            x.append(i)
            y.append(j)
            d.append(otdistance)
    dataset = DistanceDataset(torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long),  torch.tensor(d))
    dataloader = DataLoader(dataset, batch_size=batchsz, shuffle=True)
    optimizer = torch.optim.SGD(tree.parameters(), lr = lr)
    if not train_weights:
        tree.train(dataloader, optimizer, train_distributions, max_iterations=200, plt_name='/data/sam/test', bsz=1)
    else:
        tree.train_weights(dataloader,optimizer,train_distributions,max_iterations=200, plt_name='/data/sam/test', bsz=1)

    return tree, D

def simple_tree(eps = 1):
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 3, 4, 5, 6])
    G.add_edge(0, 4, weight=1)
    G.add_edge(3, 4, weight=1)
    G.add_edge(1, 5, weight=1)
    G.add_edge(2, 5, weight=1)
    G.add_edge(4, 6, weight=eps)
    G.add_edge(5, 6, weight=eps)
    x = 2 + 2 * eps
    dist =[[0, x, x, 2], 
           [x, 0, 2, x], 
           [x, 2, 0, x], 
           [2, x, x, 0]]
    return G, np.array(dist)

def calc_difference_matrix(D1, D2):
    return 2/(D1.shape[0]*(D1.shape[0]-1))*np.linalg.norm(D1 - D2, ord='fro')

# if the same as simple tree -> return True
# else, return false
def check_tree_structure(parents):
    if parents[0] == parents[3] and parents[1] == parents[2]:
        return True
    return False

def reconstruct(simple_tree, dist, lr, eps, iterations = 100, ndists=16,device='cuda:1'):
    correct_top = 0
    diff_lt = []
    for i in range(iterations):
        train_distributions = generate_random_dists(8, 4)
        learned_tree, D = tree_reconstruction_task(G=simple_tree,train_distributions=train_distributions, num_dists=ndists, lr=lr, eps=eps, device=device)
        check_structure = check_tree_structure(learned_tree.parents)
        if check_structure:
            correct_top += 1
        else:
            weights = learned_tree.param.detach().cpu().numpy()
            learned_distance = weights[learned_tree.M_idx]
            diff_lt.append(calc_difference_matrix(dist, learned_distance))

    return correct_top, diff_lt

def simple_graph_reconstruction(device, ndist=16):
    # tests with respect to learning rate, epsilon
    lrs = [0.001, 0.01, 0.1]
    noise = [0.1]
    eps_lst = [1, 0.1, 0.01, 0.001]
    lr_noise_pairs = []
    for i in range(len(lrs)):
        for j in range(len(noise)):
            lr_noise_pairs.append((lrs[i], noise[j]))

    dictionary = {}
    for eps in eps_lst:
        print("-------------------------------------")
        print("Epsilon:", eps)
        correct = []
        lt = []
        nw = []
        G, dist=simple_tree(eps = eps)
        dct = {}
        for pair in tqdm(lr_noise_pairs):
            lr = pair[0]
            gamma = pair[1]
            correct_top, diff_lt = reconstruct(G, dist, lr=lr, eps=gamma, ndists=ndist, iterations=100, device=device)
            correct.append(correct_top)
            
            if len(diff_lt) != 0:
                ltpair = (np.mean(diff_lt), np.std(diff_lt))
                lt.append(ltpair)
                dct[pair] = {'correct': correct_top, 'diff_lt': ltpair}
            else:
                lt.append(None)
                dct[pair] = {'correct': correct_top, 'diff_lt': (0, 0)}
        dictionary[eps] = dct
            
        idx = np.argmax(correct)
        print("Best learning rate/noise pair for epsilon:", lr_noise_pairs[idx])
        print("Number of correct tree topology:", correct[idx])
        print("When not correct topology, LT:", lt[idx])

    with open('small_recon2.pickle', 'wb') as handle:
        pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return dictionary

def batch_calc_twd(parents, weights, subtree, A, B):
    st = subtree.cpu().numpy()
    vA = st @ A
    vB = st @ B 
    w_v = weights[parents] - weights
    return 0.5  * w_v @ np.abs(vA - vB)

def get_wasserstein_diff(parents, parameters, subtree, D, distributions):
    #distributions = generate_random_dists(10, D.shape[0])
    n_dist = len(distributions)
    ind1 = []
    ind2 = []
    for i in range(n_dist):
        for j in range(i + 1, n_dist):
            ind1.append(i)
            ind2.append(j)
    n_samples = len(ind1)

    A = distributions[ind1].transpose()
    B = distributions[ind2].transpose()
    twd = batch_calc_twd(parents, parameters, subtree, A, B)
    ground_truth = []
    for ii in trange(n_samples):
        d1 = distributions[ind1[ii]]
        d2 = distributions[ind2[ii]]

        ground_truth.append(ot.emd2(d1, d2, D))
    ground_truth = np.array(ground_truth)
    
    return np.mean(np.abs(twd - ground_truth))
    
def get_tree_distance(T, eps=0.1):
    tree = T
    n = len(list(tree.nodes))
    nodes = list(tree.nodes)
    # find all leaf nodes
    leaves = []
    for node in range(len(nodes)):
        deg = tree.degree[node]
        if deg == 1:
            leaves.append(node)
    n = len(leaves)
    
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            leaf1 = leaves[i]
            leaf2 = leaves[j]
            D[i][j] = nx.shortest_path_length(tree, source=leaf1, target=leaf2, weight='weight')
            D[j][i] = D[i][j]
    A = np.random.normal(size=(n, n)) * eps
    M = D + A@A.T
    idx = np.where(M < 0.0)
    M[idx[0], idx[1]] = 0.0
    leaf_range = np.arange(n)
    M[leaf_range, leaf_range] = 0
    return D, M

def tree_reconstruction(device, num_trees=100, eps=3):
    tree_dataset = []
    for i in range(num_trees):
        n = np.random.randint(low=20, high=40)
        tree = nx.random_tree(n=n)
        tree_dataset.append(tree)
    
    diffs_ult = []
    diffs_wo = []
    wdiff_u = []
    wdiff_wo = []
    for i in trange(num_trees):
        T = tree_dataset[i]
        D, M = get_tree_distance(T, eps=eps)
        train_distributions = generate_random_dists(64, D.shape[0])
        test_distributions = generate_random_dists(20, D.shape[0])

        tree, D = tree_reconstruction_task(M, D_gds=D, train_distributions = train_distributions, device=device, num_dists=64, batchsz=256, eps=eps)
        weights = tree.param.detach().cpu().numpy()
        learned_distance = weights[tree.M_idx]
        diffs_ult.append(2/(D.shape[0]*(D.shape[0]-1))*np.linalg.norm(learned_distance - D, ord='fro'))
        wdiff_u.append(get_wasserstein_diff(tree.parents, weights, tree.subtree, D, test_distributions))

        # just training weights
        tree, D = tree_reconstruction_task(M, D_gds=D, train_distributions = train_distributions, device=device,randstart=True, num_dists=64, batchsz=256, train_weights=True, eps=eps)
        weights = tree.param.detach().cpu().numpy()
        learned_distance = weights[tree.M_idx]
        diffs_wo.append(2/(D.shape[0]*(D.shape[0]-1))*np.linalg.norm(learned_distance - D, ord='fro'))

        wdiff_wo.append(get_wasserstein_diff(tree.parents, weights, tree.subtree, D, test_distributions))

    print("Ultrametric optimization:", np.mean(diffs_ult), np.std(diffs_ult))
    print("Just weights:", np.mean(diffs_wo), np.std(diffs_wo))
    print("Wasserstein MAE, Ult.", np.mean(wdiff_u), np.std(wdiff_u))
    print("Wasserstein MAE, Just weights.", np.mean(wdiff_wo), np.std(wdiff_wo))
    return diffs_ult, diffs_wo, wdiff_u, wdiff_wo
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--small-tree', type=bool)
    parser.add_argument('--tree-reconstruction', type=bool)
    parser.add_argument('--device', type=str)

    args = parser.parse_args()
    print("starting small tree perturbation learning")
    if args.small_tree:
        simple_graph_reconstruction(args.device)
    print("starting tree recon.")
    if args.tree_reconstruction:
        ultD = []
        no_ultD = []
        ultW = []
        no_ultW = []
        for i in range(2, 6):
            a, b, c, d = tree_reconstruction(args.device, eps=i)
            ultD.append(a)
            no_ultD.append(b)
            ultW.append(c)
            no_ultW.append(d)
        
        for i in range(2, 6):
            print("Gaussian noise multiplier:", i)
            print("Ultrametric optimization:", np.mean(ultD[i]), np.std(ultD[i]))
            print("Just weights:", np.mean(no_ultD[i]), np.std(no_ultD[i]))
            print("Wasserstein MAE, Ult.", np.mean(ultW[i]), np.std(ultW[i]))
            print("Wasserstein MAE, Just weights.", np.mean(no_ultW[i]), np.std(no_ultW[i]))


if __name__=='__main__':
    main()