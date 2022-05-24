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
import os
import multiprocessing as mp
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from mst import scipy_mst, gpu_mst
import pickle
import dill

def run_dill_encoded(payload):
    fun, args = dill.loads(payload)
    return fun(*args)

def apply_async(pool, fun, args):
    payload = dill.dumps((fun, args))
    return pool.apply_async(run_dill_encoded, (payload,))



def generate_distance_metric(points):
    D = np.zeros((len(points), len(points)))
    pool = mp.Pool(processes=20)
    jobs = []
    for i in trange(len(points)):
        for j in range(i + 1, len(points)):
            job = pool.apply_async(distance.euclidean, args=(points[i], points[j]))
            jobs.append(job)
            
    for job in tqdm(jobs):
        job.wait()
    results = [job.get() for job in jobs]
    count = 0
    for i in trange(len(points)):
        for j in range(i + 1, len(points)):
            result = results[count]
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

# use nn.Module
class GPUOptimizedTree(nn.Module):
    def __init__(self, distance_matrix, device="cpu"):
        super(GPUOptimizedTree, self).__init__()
        self.device = device
        self.D = distance_matrix
        self.n = distance_matrix.shape[0]
        self.triu_indices = np.triu_indices(self.n, 1)
        self.vec = self.D[self.triu_indices]
        self.np_parameters = np.zeros((2*self.n - 1), dtype=np.double)
        self.leaves = np.arange(0, self.n, dtype=np.int32)
        self.parents = np.zeros((2*self.n - 1), dtype=np.int32)
        self.M_idx = np.zeros((self.n, self.n), dtype=np.int32)
        self.subtree = gpu_mst(self.vec, self.n, self.M_idx, self.np_parameters, self.leaves, self.parents)
        self.subtree = torch.tensor(self.subtree.astype(np.double), device = self.device)
        #print(self.subtree)
        
        self.param = nn.Parameter(torch.tensor(self.np_parameters, device = self.device), requires_grad=True)    
    
    
    # compute tree wasserstein distance 
    def wasserstein1(self, d1, d2):
        m_d1 = torch.matmul(self.subtree, d1)[:2 * self.n - 2]
        m_d2 = torch.matmul(self.subtree, d2)[:2 * self.n - 2]
        w_v = self.param[self.parents[:2*self.n - 2]] - self.param[:2*self.n - 2]
        return 0.5 * torch.matmul(w_v,torch.abs(m_d1 - m_d2))

    # 0 for similar
    # 1 for dis-similar
    def calc_contrastive_loss(self, d1, d2, labels, m=1):
        mask = labels != 0
        
        ds = (1/n_ds) * torch.sum(torch.min(self.wasserstein1(d1[mask], d2[mask]), m))
        s = (1/n_s) * torch.sum(self.wasserstein1(d1[-mask], d2[-mask]))
        return s - ds

    # compute wasserstein loss 
    def calc_wasserstein1_loss(self, d1, d2, ot_distances):
        
        m_d1 = torch.matmul(self.subtree, d1)[:2 * self.n - 2]
        m_d2 = torch.matmul(self.subtree, d2)[:2 * self.n - 2]
        w_v = self.param[self.parents[:2*self.n - 2]] - self.param[:2*self.n - 2]
        w1_tree = 0.5 * torch.matmul(w_v, torch.abs(m_d1- m_d2))
        #print(m_d1 - m_d2)
        return torch.sum(torch.square(ot_distances - w1_tree))**0.5


    def train(self, dataloader, optimizer, train_distributions, max_iterations=5, plt_name="test_losses", save_epoch=False):        
        losses = []        
        batch_losses = []
        num_batches = len(dataloader)
        for i in trange(max_iterations):
            batch_loss = 0.0
            for batch in tqdm(dataloader):
                optimizer.zero_grad()
                b1_dist = torch.tensor(train_distributions[batch[0]])
                b2_dist = torch.tensor(train_distributions[batch[1]])
                b1 = torch.transpose(b1_dist, 0, 1).to(self.device)
                b2 = torch.transpose(b2_dist, 0, 1).to(self.device)
                ot_dist = batch[2].to(self.device)
                start = time.time()
                loss = self.calc_wasserstein1_loss(b1, b2, ot_dist)
                end = time.time()
                print("Time per batch:", end - start)
                batch_loss += loss.item()
                losses.append(loss.item())
                loss.backward()
                optimizer.step()

                
                self.np_parameters = self.param.detach().cpu().numpy().astype(np.double)
                self.vec = self.np_parameters[self.M_idx[self.triu_indices]]
                start =  time.time()
                self.subtree = scipy_mst(self.vec, self.n, self.M_idx, self.np_parameters, 
                                        self.leaves, self.parents)
                end = time.time()
                print("Time for MST:", end - start)
                self.subtree = torch.tensor(self.subtree.astype(np.double), device = self.device)
                #self.subtree.to(device)
                with torch.no_grad():
                    for j in range(2*self.n - 1):
                        if j < self.n:
                            self.param[j] = 0.0
                            # self.parameters[j] = torch.min(self.np_parameters[M[j]])
                        else:
                            self.param[j] = self.np_parameters[j]
                # print(self.np_parameters)
                # print(self.parameters)
            batch_losses.append(batch_loss/num_batches)
            if len(batch_losses) >=10 and (abs(batch_losses[-1] - batch_losses[-2]) < 0.5):
                print("Trained for", i, "epochs")
                break
        plt.plot(np.arange(0, len(losses)), losses)
        plt.savefig(plt_name)


# Tree with C++ tree Wasserstein computation. 
class OptimizedTree:
    
    def __init__(self, distance_matrix):
        self.D = distance_matrix
        self.n = distance_matrix.shape[0]
        print("NUMBER OF LEAVES:", self.n)

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
        self.subtree = scipy_mst(self.vec, self.n, self.M_idx, self.np_parameters, self.leaves, self.parents)
        self.subtree = torch.tensor(self.subtree.astype(np.double))
        self.parameters = nn.Parameter(torch.tensor(self.np_parameters))
        self.solver = te.TreeEstimators()
        self.solver.load_tree(self.parents, self.leaves)

        self.parent_param_index = np.zeros((2*self.n - 2), dtype=np.int32)
        for i in range(1, 2*self.n - 1):
            if self.parents[i] != 0:
                self.parent_param_index[i - 1] = self.parents[i] - 1
            else:
                self.parent_param_index[i - 1] = 2 * self.n - 2
        
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
        optimizer.to(device)

        losses = []

        for i in trange(max_iterations):
            for batch in dataloader:
                optimizer.zero_grad()
                self.solver.load_tree(self.parents, self.leaves)
                mu_idx = batch[0]
                rho_idx = batch[1]
                mu_batchf = mus_f[mu_idx]
                rho_batchf = mus_f[rho_idx]
                ot_distances = batch[2]
#                 for k in range(len(mu_idx)):
#                     mu_batchf.append(mus_f[mu_idx[k].item()])
#                     rho_batchf.append(mus_f[rho_idx[k].item()])
                start = time.time()
                loss = self.calc_wasserstein_loss(mu_batchf, rho_batchf, ot_distances)
                losses.append(loss.item())
                loss.backward()
                #print(self.parameters.grad)
                
                optimizer.step()
                
                self.np_parameters = self.parameters.detach().numpy().astype(np.double)
                self.vec = self.np_parameters[self.M_idx[self.triu_indices]]
                scipy_mst(self.vec, self.n, self.M_idx, self.np_parameters, self.leaves, self.parents)

                with torch.no_grad():
                    for j in range(2*self.n - 1):
                        self.parameters[j] = self.np_parameters[j]         
            if save_epoch:
                per_epoch_name = "t4_e" + str(i)
                self.save_tree_data("/data/sam/twitter/results/30", per_epoch_name)
            if len(losses) >=300 and (abs(losses[-1] - losses[-2]) < 0.5 or losses[-1] > losses[-2]):
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
    np.random.seed(1)
    n_points = 100
    pointset = generate_random_points(n_points)
    D = generate_distance_metric(pointset)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    tree = GPUOptimizedTree(D, device = device)
    tree.to(device)
    #D = np.array([[0, 1, 5, 3], [1, 0, 3, 5], [5, 3, 0, 1], [3, 5, 1, 0]], dtype=np.double)
    d1_train = generate_random_dists(100, n_points)
    d2_train = generate_random_dists(2, n_points)

    x = []
    y = []
    d = []
    for i in range(len(d1_train)):
        for j in range(i + 1, len(d1_train)):
            x.append(d1_train[i])
            
            y.append(d1_train[j])
            
            d.append(ot.emd2(d1_train[i], d1_train[j], D))

    dataset = DistanceDataset(torch.tensor(x), torch.tensor(y), torch.tensor(d))
    dataloader = DataLoader(dataset, batch_size=int(100*(100 - 1)*0.5))

    optimizer = torch.optim.Adam(tree.parameters(), lr=0.1)
    tree.train(dataloader, optimizer, max_iterations=50)

    # x = []
    # y = []
    # d = []
    # for i in range(len(d1_train)):
    #     for j in range(i + 1, len(d1_train)):
    #         x.append(i)
            
    #         y.append(j)
    #         d.append(ot.emd2(d1_train[i], d1_train[j], D))
    # dataset = DistanceDataset(torch.tensor(x), torch.tensor(y), d)
    # dataloader = DataLoader(dataset, batch_size = 100)
    # #D = np.array([[0, 1, 5, 3], [1, 0, 3, 5], [5, 3, 0, 1], [3, 5, 1, 0]], dtype=np.double)
    # tree1 = OptimizedTree(D)
    # tree1.train(dataloader, d1_train, max_iterations=50, plt_name="test_losses1")


if __name__=="__main__":
    test_tree()
