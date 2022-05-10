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
from utils import *
import torch.multiprocessing as tmp
import multiprocessing as mp
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from mst import scipy_mst

class DistanceDataset(Dataset):
    def __init__(self, idx, idy, d):
        self.x = idx
        self.y = idy
        self.d = d

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, id):
        return self.x[id], self.y[id], self.d[id]

class OptimizedTree:
    
    def __init__(self, distance_matrix):
        self.D = distance_matrix

        # initialize the following fields with the MST transform 
        self.parameters = torch.zeros()
        # numpy copy of parameters to use in mst_transform
        # note that this numpy copy is necessary to for cython speedup in mst
        # numpy double array
        self.np_parameters = np.zeros()
        self.leaves = np.arange()
        self.parents = np.zeros()
        self.solver = te.TreeEstimator()
        self.M_idx = np.zeros()
        self.n = distance_matrix.shape[0]

        # initialize self.vec afterwards
        self.vec = None
    
    def calc_wasserstein_loss(self, d1_f, d2_f, ot_distances):
        loss = 0.0
        for i in range(len(d1)):
            mu = d1_f
            rho = d2_f
            loss += (ot_distances[i] - tree_wasserstein(self.M_idx, self.parameters, self.solver, mu, rho))**2
        return loss**0.5

    def mst_transform():
        return 0

    def train(self, dataloader, mus, rhos, max_iterations=5, plt_name="losses"):
        # format distributions for tree solver
        mus_f = []
        rhos_f = []
        for mu in mus:
            mus_f.append(format_distributions(mu))
        for rho in rhos:
            rhos_f.append(format_distributions(rho))

        # initial parameters
        self.parameters = format_distance_matrix()
        optimizer = torch.optim.Adam([self.parameters], lr=0.1)

        loss = []

        for i in trange(max_iterations):
            for batch in dataloader:
                optimizer.zero_grad()
                solver.load_tree(self.parents, self.leaves)
                mu_idx = batch[0]
                rho_idx = batch[1]
                mu_batchf = []
                rho_batchf = []
                for k in range(len(batch1)):
                    mu_batchf.append(mus_f[mu_idx[k].item()])
                    rho_batchf.append(rhos_f[rho_idx[k].item()])
                ot_distances = batch[3]
                loss = calc_wasserstein_loss(mu_batchf, rho_batchf, ot_distances)
                loss.backward()
                optimizer.step()
                
                mst_transform()
        plt.plot(np.arange(0, len(losses)), losses)
        plt.savefig(plt_name)


# testing code
def test_tree():
    return 0


if __name__=="__main__":
    test_tree()