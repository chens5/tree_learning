import sched
import numpy as np
import networkx as nx
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
import ot
import scipy.spatial.distance as distance
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import *
from tqdm import trange, tqdm
import time
from utils import *
import multiprocessing as mp
from torch.utils.data import Dataset
from mst import scipy_mst, gpu_mst
from scipy.cluster.hierarchy import linkage

def generate_distances(points):
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
    return results

# Generate distance matrix for a set of points. 
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

# Distance dataset. 
class DistanceDataset(Dataset):
    def __init__(self, idx, idy, d):
        self.x = idx
        self.y = idy
        self.d = d

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, id):
        return self.x[id], self.y[id], self.d[id]
    
# Optimizing ultrametric trees via the torch nn.Module superclass. 
# Note that this tree can be saved by calling torch.save and saving
# the state_dict of the model as well as the leaves and parents of the
# tree. 
class UltrametricOptimizedTree(nn.Module):
    def __init__(self, distance_matrix, device="cpu"):
        super(UltrametricOptimizedTree, self).__init__()
        self.device = device
        self.D = distance_matrix
        self.n = distance_matrix.shape[0]
        self.triu_indices = np.triu_indices(self.n, 1)
        self.vec = self.D[self.triu_indices].astype(np.double)
        self.np_parameters = np.zeros((2*self.n - 1), dtype=np.double)
        self.leaves = np.arange(0, self.n, dtype=np.int32)
        self.parents = np.zeros((2*self.n - 1), dtype=np.int32)
        self.M_idx = np.zeros((self.n, self.n), dtype=np.int32)

        self.subtree = gpu_mst(self.vec, self.n, self.M_idx, self.np_parameters, self.leaves, self.parents)
        self.subtree = torch.tensor(self.subtree.astype(np.double), device = self.device)
        
        self.param = nn.Parameter(torch.tensor(self.np_parameters, device = self.device), requires_grad=True)    
        self.initial_tree = np.copy(self.parents)
    

    def compute_flow(self, d1, d2):
        matching = np.zeros((self.n, self.n))
        current = {}
        leaf_queue = np.copy(self.leaves)
        parents = np.copy(self.parents)
        # Fill current with leaf node weights
        

        while not leaf_queue:
            node_id = leaf_queue.pop()
            parent = parents[node_id]
            # subtract weights in node
            d1weights, d2weights = current[node_id]
            while d1weights and d2weights:
                # tuple of leaf_node id and weight
                id1 = d1weights[0][0]
                weight1 = d1weights[0][1]

                id2 = d1weights[0][0]
                weight2 = d1weights[0][1]

                matched_weight = min(weight1, weight2)
                matching[id1, id2] = matched_weight

                if weight1 - matched_weight == 0:
                    d1weights.pop()
                if weight2 - matched_weight == 0:
                    d2weights.pop()

            # push remaining weight and leaf identity up to next level
            if d1weights:
                current[parent][0].append(d1weights)
            if d2weights:
                current[parent][1].append(d2weights)
            # remove node from parent list 
            parents[node_id] = -1
            # check if there is another node in the parent list which has its parent
            is_inner = parent in parents
            # if no, push node id into queue
            if not is_inner:
                leaf_queue.append(parent)

        return matching

    # compute tree wasserstein distance 
    def wasserstein1(self, d1, d2):
        m_d1 = torch.matmul(self.subtree, d1)[:2 * self.n - 2]
        m_d2 = torch.matmul(self.subtree, d2)[:2 * self.n - 2]
        w_v = self.param[self.parents[:2*self.n - 2]] - self.param[:2*self.n - 2]
        return 0.5 * torch.matmul(w_v,torch.abs(m_d1 - m_d2))

    # compute wasserstein loss 
    def calc_wasserstein1_loss(self, d1, d2, ot_distances):
        
        m_d1 = torch.matmul(self.subtree, d1)
        m_d2 = torch.matmul(self.subtree, d2) 
        # self.param[self.parents] get weights of node parents
        # self.param weight of node themselves
        # thus w_v = weight of edge from a node (represented by the coordinate) to parent
        w_v = self.param[self.parents] - self.param
        w1_tree = 0.5 * torch.matmul(w_v, torch.abs(m_d1 - m_d2))
        del m_d1
        del m_d2
        del w_v
        return torch.sum(torch.square(ot_distances - w1_tree))


    def train(self, dataloader, optimizer, train_distributions, 
              max_iterations=5, plt_name="test_losses", save_epoch=False, 
              filename='/data/sam/model.pt', bsz=-1):        
        losses = []        
        train_distributions = torch.tensor(train_distributions).to(self.device)
        optimizer.zero_grad()
        time_p_epoch = []
        for i in trange(max_iterations):
            loss = 0.0
            
            total = 0
            num_batches = 0
            start = time.time()
            for batch in dataloader:
                b1_dist = train_distributions[batch[0]]
                b2_dist = train_distributions[batch[1]]
                total += len(batch[0])
                # b1_dist = batch[0]
                # b2_dist = batch[1]
                b1 = torch.transpose(b1_dist, 0, 1)
                b2 = torch.transpose(b2_dist, 0, 1)
                ot_dist = batch[2].to(self.device)
                loss = self.calc_wasserstein1_loss(b1, b2, ot_dist)
                
                loss.backward()   
                num_batches += 1     
                if bsz > 0.0 and num_batches % bsz == 0:
                    losses.append(loss.detach().cpu())                    
                    #torch.nn.utils.clip_grad_norm_(self.param, 0.5)
                    optimizer.step()

                    self.np_parameters = self.param.detach().cpu().numpy().astype(np.double)
                    new_distances = 0.5*(2 * self.np_parameters[self.M_idx] - self.np_parameters[:self.n][:, None]  - self.np_parameters[:self.n])
                    self.vec = new_distances[self.triu_indices]

                    self.subtree = gpu_mst(self.vec, self.n, self.M_idx, self.np_parameters, 
                                            self.leaves, self.parents)
                    #self.mst()
                    self.subtree = torch.tensor(self.subtree.astype(np.double), device = self.device)
                    
                    with torch.no_grad():
                        for j in range(2*self.n - 1):
                            if j < self.n:
                                self.param[j] = 0.0
                                #self.parameters[j] = torch.min(self.np_parameters[self.M_idx[j]])
                            elif self.param[j] < 0.0:
                                self.param[j] = 0.0
                            else:
                                self.param[j] = self.np_parameters[j]
                    optimizer.zero_grad()

            end = time.time()
            time_p_epoch.append(end - start)
            if save_epoch:
                sf = filename + str(i) + '.pt'
                torch.save({'parents': self.parents,
                'leaves': self.leaves,
                'subtree':self.subtree,
                'vec': self.vec,
                'M_idx': self.M_idx,
                'np_parameters': self.np_parameters,
                'optimized_state_dict': self.state_dict()}, 
                sf)
            if len(losses ) >=7 and abs(losses[-1] - losses[-6]) <= 0.01:
                break
        print("Average time per epoch", np.mean(time_p_epoch), "standard deviation,", np.std(time_p_epoch))
        plt.plot(np.arange(0, len(losses)), losses)
        plt.savefig(plt_name)
    

# testing code
def test_tree():
    pointset = np.array([[0, 0], [0, 1], [10, 0], [10, 1]])
    D = generate_distance_metric(pointset)
    tree = UltrametricOptimizedTree(D, device = "cpu")
    print(tree.subtree)
    print(tree.param)
    A = np.array([[0.0, 0.25, 0.25, 0.50], [0.5, 0.5, 0.0, 0.0]])
    B = np.array([[0.25, 0.5, 0.25, 0.0], [0.5, 0.25, 0.0, 0.25]])
    ot.emd2(A[0], B[0], D)
    ot.emd2(A[1], B[1], D)
    d = torch.tensor([ot.emd2(A[0], B[0], D), ot.emd2(A[1], B[1], D)])
    tree.calc_wasserstein1_loss(torch.transpose(torch.tensor(A), 0, 1), torch.transpose(torch.tensor(B), 0, 1), d)
    return
    


if __name__=="__main__":
    test_tree()
