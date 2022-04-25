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
from utils import *

def train_ultrametric(D, dists1, dists2, max_iterations=5):
    fig, axs = plt.subplots(1, figsize=(10, 10))

    M1 = D.copy()
    M2 = D.copy()
    parameters1 = format_distance_matrix(M1)

    parameters2 = format_distance_matrix(M2)

    optimizer = torch.optim.Adam([parameters1, parameters2], lr=0.01)
    M1 = M1.astype(int)
    M2 = M2.astype(int)
    # initial tree

    tree, parents, p, leaves, root = mst_transform(M1, M2, parameters1, parameters2)
    solver = te.TreeEstimators()
    solver.load_tree(p, leaves)
    print("Root 1:", root)

    losses = []
    ultramatrix_ref = None
    values = None
    for i in trange(max_iterations):
        optimizer.zero_grad()
        solver.load_tree(p, leaves)
        if i % 2 == 0:
            loss = loss_fn(M2, parameters2, solver, D, dists1, dists2)
        else:
            loss = loss_fn(M1, parameters1, solver, D, dists1, dists2)

        losses.append(loss.detach())
        loss.backward()
        optimizer.step()

        if i % 2 == 0:
            tree, parents, p, leaves, root = mst_transform(M2, M1, parameters2, parameters1)
            ultramatrix_ref = M1
            values = parameters1
        else:
            tree, parents, p, leaves, root = mst_transform(M1, M2, parameters1, parameters2)
            ultramatrix_ref = M2
            values = parameters2

    print("beginning loss:", losses[0])
    print("ending loss:", losses[max_iterations-1])
    print("Root end:", root)
    return tree, parents, p, leaves, root, M2, parameters2
