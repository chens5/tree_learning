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
import torch.multiprocessing as tmp
import multiprocessing as mp
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from mst import scipy_mst

def generate_distance_metric(points):
    D = np.zeros((len(points), len(points)))
    pool = mp.Pool(processes=20)
    jobs = []
    for i in trange(len(points)):
        for j in range(i + 1, len(points)):
            job = pool.apply_async(distance.euclidean, args=(points[i], points[j]))
            jobs.append(job)
            
    for job in tqdm(jobs):
        job.wait
    #results = [job.get() for job in jobs]
    count = 0
    for i in trange(len(points)):
        for j in range(i + 1, len(points)):
            result = jobs[count].get()
            D[i][j] = result
            D[j][i] = result
            count += 1
    return D

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
        self.n = distance_matrix.shape[0]

        self.triu_indices = np.triu_indices(self.n, 1)

        # initialize self.vec from distance matrix
        self.vec = self.D[self.triu_indices]

        # initialize the following fields with the MST transform 
        # note that this numpy copy of parameters is necessary for cython speedup in mst
        # numpy double array
        self.np_parameters = np.zeros((2*self.n - 1), dtype=np.double)
        self.leaves = np.arange(1, self.n + 1, dtype=np.int32)
        self.parents = np.zeros((2*self.n - 1), dtype=np.int32)
        self.M_idx = np.zeros((self.n, self.n), dtype=np.int32)
        scipy_mst(self.vec, self.n, self.M_idx, self.np_parameters, self.leaves, self.parents)
        self.parameters = nn.Parameter(torch.tensor(self.np_parameters))
        self.solver = te.TreeEstimators()
        self.solver.load_tree(self.parents, self.leaves)
        
    def save_tree_data(self, folder, exp_name):
        lst_names = ["_parents.npy", "_leaves.npy", "_parameters.npy", "_Midx.npy"]
        data = [self.parents, self.leaves, self.np_parameters, self.M_idx]
        for i in range(4):
            fname = folder + "/" + exp_name + lst_names[i]
            np.save(fname, data[i])
    

    def calc_wasserstein_loss(self, d1_f, d2_f, ot_distances):
        loss = 0.0
        for i in range(len(d1_f)):
            mu = d1_f[i]
            rho = d2_f[i]
            tree_wasserstein(self.M_idx, self.parameters, self.solver, mu, rho)
            loss += (ot_distances[i] - tree_wasserstein(self.M_idx, self.parameters, self.solver, mu, rho))**2
        return loss**0.5


    def train(self, dataloader, distributions, max_iterations=5, plt_name="losses", save_epoch=False):
        # format distributions for tree solver
        mus_f = []
        for mu in distributions:
            mus_f.append(format_distributions(mu))
        mus_f=np.array(mus_f, dtype=object)
        # initial parameters
        optimizer = torch.optim.Adam([self.parameters], lr=0.1)

        losses = []

        for i in trange(max_iterations):
            for batch in dataloader:
                optimizer.zero_grad()
                self.solver.load_tree(self.parents, self.leaves)
                mu_idx = batch[0]
                rho_idx = batch[1]
                mu_batchf = mus_f[mu_idx]
                rho_batchf = mus_f[rho_idx]
#                 for k in range(len(mu_idx)):
#                     mu_batchf.append(mus_f[mu_idx[k].item()])
#                     rho_batchf.append(mus_f[rho_idx[k].item()])
                ot_distances = batch[2]

                loss = self.calc_wasserstein_loss(mu_batchf, rho_batchf, ot_distances)
                losses.append(loss.item())
                loss.backward()
                
                optimizer.step()
                
                self.np_parameters = self.parameters.detach().numpy().astype(np.double)
                self.vec = self.np_parameters[self.M_idx[self.triu_indices]]
                scipy_mst(self.vec, self.n, self.M_idx, self.np_parameters, self.leaves, self.parents)

                with torch.no_grad():
                    for j in range(2*self.n - 1):
                        self.parameters[j] = self.np_parameters[j]         
            if save_epoch:
                per_epoch_name = "t2_e" + str(i)
                self.save_tree_data("/data/sam/twitter/results/30", per_epoch_name)
            if len(losses) >=10 and (abs(losses[-1] - losses[-2]) < 0.5 or losses[-1] > losses[-2]):
                print("Trained for", i, "epochs")
                break
        plt.plot(np.arange(0, len(losses)), losses)
        plt.savefig(plt_name)


class WeightOptimizedQuadtree:
    def __init__(self, num_points):
        self.solver = ote.OTEstimators()
        self.solver.load_vocabulary(pointset)
        self.num_nodes = len(self.solver.return_parent_details())
        self.parameters = nn.Parameter(torch.ones(self.num_nodes))
    
    # compute embedding from raw vector
    def compute_embedding(self, a):
        return self.solver.compute_raw_vector(a)
    
    def wasserstein(self, v_a, v_b):

        # dot product self.parameters and |v_a - v_b|
        return 0 
    
    # each row represents vector embedding
    def wasserstein_loss(self, v_a, v_b, ot_distances):
        loss = 0.0
        return torch.sqrt(torch.sum(torch.square(ot_distances - np.abs(v_a - v_b) @ self.parameters)))
    
    def train(self, dataloader,lr=0.01, max_iterations=5, plt_name="qt_losses", save_epoch=False):
        optimizer = torch.optim.Adam([self.parameters], lr=lr)
        for i in trange(max_iterations):
            for batch in dataloader:
                optimizer.zero_grad()
                loss = self.wasserstein_loss(batch[0], batch[1], batch[2])
                losses.append(loss)
                optimizer.step()
                
        plt.plot(np.arange(0, len(losses)), losses)
        plt.savefig(plt_name)
        return 0



# testing code
def test_tree():
    np.random.seed(0)
    n_points = 10
    pointset = generate_random_points(n_points)
    D = generate_distance_metric(pointset)
    d1_train = generate_random_dists(5, n_points)
    d2_train = generate_random_dists(5, n_points)

    x = []
    y = []
    d = []
    for i in range(len(d1_train)):
        for j in range(len(d2_train)):
            x.append(i)
            y.append(j)
            d.append(ot.emd2(d1_train[i], d1_train[j], D))
    dataset = DistanceDataset(torch.tensor(x), torch.tensor(y), d)
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
    tree = OptimizedTree(D)
    tree.train(dataloader, d1_train, d2_train)


if __name__=="__main__":
    test_tree()
