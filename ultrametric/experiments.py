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
from mst import mst

# python wrapper for c++ class
class TreeEstimator:
    def __init__(self):
        self.solver = te.TreeEstimators()
        
    def load_tree(self, p, leaves):
        self.solver.load_tree(p, leaves)
        
    def tree_query(self, mu, rho):
        self.solver.tree_query(mu, rho)
    
    def return_matching(self):
        return self.solver.return_matching()
    
    def return_mass(self):
        return self.solver.return_mass()

def compare_avg_error(pointset,D, leaves, parents, M_idx, parameters):
    d1_test = generate_random_dists(50, len(pointset))
    d2_test = generate_random_dists(50, len(pointset))
    errors_tw = []
    tree_solver = TreeEstimator()
    tree_solver.load_tree(parents, leaves)
    params = parameters.detach().numpy()
    
    solver = ote.OTEstimators()
    solver.load_vocabulary(pointset)
    print("Successfully loaded QT")
    errors_qt = []
    errors_ft = []
    
    for i in range(len(d1_test)):
        for j in range(len(d2_test)):
            d1 = format_distributions(d1_test[i])
            d2 = format_distributions(d2_test[i])
            tw = calc_tree_wasserstein(M_idx, params, tree_solver, d1, d2)
            qt = solver.quadtree_distance(d1, d2)
            ft_distance = solver.flowtree_query(d1, d2)
            w1_distance = ot.emd2(d1_test[i], d2_test[i], D)
            errors_tw.append(abs(tw - w1_distance))
            errors_qt.append(abs(w1_distance- qt))
            errors_ft.append(abs(ft - w1_distance))
    return errors_tw, errors_qt, errors_ft
    
def compare_nn_plots(pointset,D, leaves, parents, M_idx, parameters, dim):
    queries = generate_random_dists(100, len(pointset))
    candidates = generate_random_dists(2000, len(pointset))
    tree_solver = TreeEstimator()
    tree_solver.load_tree(parents, leaves)
    params = parameters.detach().numpy()
    
    solver = ote.OTEstimators()
    solver.load_vocabulary(pointset)
    print("Successfully loaded QT")

    #np.save("syntheticR2_parents1", p)
    #np.save("syntheticR2_leaves1", leaves)
    distances = {'q':[], 'c':[], 'w1':[], 'qt':[], 'ft':[], 'tw':[]}
    for i in trange(len(queries)):
        for j in range(len(candidates)):
            distances['q'].append(i)
            distances['c'].append(j)
            d1 = format_distributions(queries[i])
            d2 = format_distributions(candidates[j])
            distances['w1'].append(ot.emd2(queries[i], candidates[j], D))
            distances['qt'].append(solver.quadtree_distance(d1, d2))
            distances['ft'].append(solver.flowtree_query(d1, d2))
            distances['tw'].append(tree_wasserstein(M, params, tree_solver, d1, d2).item())
        
    df = pd.DataFrame.from_dict(distances)
    tw = [0]*100
    qt = [0]*100
    ft = [0]*100
    for i in trange(len(queries)):
        query_d = df['q'] == i
        filtered_data = df[query_d]
        id_true = filtered_data['c'][filtered_data['w1'].idxmin()]
        for m in range(1, 101):
            top_m_ft = filtered_data.nsmallest(m, 'ft')
            top_m_qt = filtered_data.nsmallest(m, 'qt')
            top_m_tw = filtered_data.nsmallest(m, 'tw')
            if id_true in list(top_m_ft['c']):
                ft[m - 1] += 1
            if id_true in list(top_m_qt['c']):
                qt[m - 1] += 1
            if id_true in list(top_m_tw['c']):
                tw[m - 1] += 1
    plt.clf()
    plt.plot(np.arange(1, 101), tw, label="Optimized tree")
    plt.plot(np.arange(1, 101), qt, label="QT")
    plt.plot(np.arange(1, 101), ft, label="FT")
    plt.legend()
    name = "nn_dim"+str(dim)
    plt.savefig(name)
    
def calc_tree_wasserstein(UM, parameters, solver, mu, rho):
    solver.tree_query(mu, rho)
    matching = solver.return_matching()
    masses = solver.return_mass()
    return torch.sum(torch.tensor(masses)*parameters[UM[matching[0], matching[1]]])


def calc_loss(dist1, dist2, dist1_f, dist2_f, distances, solver, M_idx, parameters):
    loss = 0.0
    jobs = []
    for i in range(len(dist1)):
        mu = dist1_f[i]
        rho = dist2_f[i]
        loss += (distances[i] - tree_wasserstein(M_idx, parameters, solver, mu, rho))**2
    return (loss)**0.5

def mst_transform(D, leaves, parents):
    vec = np.triu_indices(D, 1)
    np_params = parameters.detach.numpy().astype(np.double)
    parents = scipy_mst(vec, n, M_idx, parameters, leaves, parents)
    with torch.no_grad():
        parameters = np_params
    return np_params

def train_ultrametric(dataloader, D, dists1, dists2, max_iterations=5, dim=2):
    #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    #fig, axs = plt.subplots(1, figsize=(10, 10))

    M1 = D.copy()
    M2 = D.copy()
    parameters1 = format_distance_matrix(M1)

    parameters2 = format_distance_matrix(M2)
    formatted1 =[]
    formatted2 = []
    for dist in dists1:
        formatted1.append(format_distributions(dist))
    for dist in dists2:
        formatted2.append(format_distributions(dist))
    optimizer = torch.optim.Adam([parameters1, parameters2], lr=0.1)
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
        for batch in dataloader:
            optimizer.zero_grad()
            solver.load_tree(p, leaves)
            #batch1 = batch[0].to(device)
            #batch2 = batch[1].to(device)
            #dist = batch[2].to(device)
            batch1= batch[0]
            batch2 = batch[1]
            batch1_idx = batch[2]
            batch2_idy = batch[3]
            batch1_f = []
            batch2_f = []
            for k in range(len(batch1)):
                batch1_f.append(formatted1[batch1_idx[k].item()])
                batch2_f.append(formatted2[batch2_idy[k].item()])
            dist = batch[4]
            
            if i % 2 == 0:
                loss = calc_loss(batch1, batch2, batch1_f, batch2_f, dist, solver, M2, parameters2)
            else: 
                loss = calc_loss(batch1, batch2, batch1_f, batch2_f, dist, solver, M1, parameters1)

            loss.backward()
            optimizer.step()
            if i % 2 == 0:
                tree, parents, p, leaves, root =mst_transform(M2, M1, parameters2, parameters1)
                ultramatrix_ref = M1
                values = parameters1
            else:
                tree, parents, p, leaves, root = mst_transform(M1, M2, parameters1, parameters2)
                ultramatrix_ref = M2
                values = parameters2
            losses.append(loss.item())
    plt.plot(np.arange(0, len(losses)), losses, label="dim="+str(dim))
    return p, leaves, root, ultramatrix_ref, values

class DistanceDataset(Dataset):
    def __init__(self, x, y, x_f, y_f, d):
        self.x = x
        self.y = y
        self.d = d
        self.x_f = x_f
        self.y_f = y_f

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.x_f[idx], self.y_f[idx], self.d[idx]
    
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
    
def dim_experiment(sz, dims):
    all_leaves = []
    all_parents = []
    M_idxs = []
    values = []
    pointsets = []
    for k in range(len(dims)):
        n_points = sz
        pointset = generate_random_points(n_points, dim=dims[k], low=-100, high=100)
        pointset = pointset.astype(np.float32)
        #print(pointset)
        D = generate_distance_metric(pointset)
        d1_train = generate_random_dists(50, n_points)
        d2_train = generate_random_dists(50, n_points)

        x = []
        y = []
        d = []
        for i in range(len(d1_train)):
            for j in range(len(d2_train)):
                x.append(d1_train[i])
                y.append(d2_train[i])
                d.append(ot.emd2(d1_train[i], d1_train[j], D))
        dataset = DistanceDataset(torch.tensor(x), torch.tensor(y), d)
        dataloader = DataLoader(dataset, batch_size=500, shuffle=True)

        ## train tree structure
        p, leaves, root, M, parameters=train_ultrametric(dataloader, D, d1_train, d1_train, max_iterations=100, dim=dims[k])
        all_leaves.append(leaves)
        all_parents.append(p)
        M_idxs.append(M)
        values.append(parameters)
        pointsets.append(pointset)
    plt.legend()
    plt.savefig("loss_p_iter_DIM")
    for i in range(len(dims)):
        D = generate_distance_metric(pointsets[i])
        tw, qt, ft = compare_avg_error(pointsets[i], D, all_leaves[i], all_parents[i], M_idxs[i], values[i])
        print("Dimension=", dims[i])
        print("Average TW error:", np.mean(tw), "std dev.", np.std(tw))
        print("Average QT error:", np.mean(qt), "std dev.", np.std(qt))
        print("Average FT error:", np.mean(ft), "std dev.", np.std(ft))
        compare_nn_plots(pointsets[i], D, all_leaves[i], all_parents[i], M_idxs[i], values[i], dims[i])

def document_classification():
    word_vecs = np.load('data/twitter_vecs.npy')
    documents = np.load('data/twitter_dists.npy')
    labels = np.load('data/twitter_labels.npy')
    #D = generate_distance_metric(word_vecs)
    #np.save("distance_matrix_twitter.npy", D)
    D = np.load("distance_matrix_twitter.npy")
    d1_train = documents[:10]
    d2_train = documents[10:20]
    x = []
    y = []
    d = []
    idx = []
    idy = []
    for i in range(len(d1_train)):
        for j in range(len(d2_train)):
            x.append(d1_train[i])
            y.append(d2_train[j])
            idx.append(i)
            idy.append(j)
            d.append(ot.emd2(d1_train[i], d2_train[j], D))
    dataset = DistanceDataset(torch.tensor(x), torch.tensor(y), idx, idy, d)
    dataloader = DataLoader(dataset, batch_size=200, shuffle=True)


    
    
def main():
    #dims = [10, 20, 30, 50, 60, 70]
    #dim_experiment(100, dims)
    document_classification()
        
    
if __name__=="__main__":
    main()


