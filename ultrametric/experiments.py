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
from ot_tree import *
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score
import sys
from scipy.stats import mode

def document_classification(dataset):
    wv_fname = '/data/sam/' + dataset+ '/vectors.npy'
    word_vecs = np.load(wv_fname)
    print("Number of words in BBC dataset", len(word_vecs)) 
    document_fname = '/data/sam/' + dataset + '/distributions.npy'
    documents = np.load(document_fname)
    lbl_fname = '/data/sam/' + dataset + '/labels.npy'
    labels = np.load(lbl_fname)
    print("Generating distance matrix.....")
    D = generate_distance_metric(word_vecs)
    distance_mat_name = '/data/sam/' + dataset + '/distance_matrix.npy'
    np.save(distance_mat_name, D)
    #D = np.load("/data/sam/bbc/distance_matrix.npy")

    X_train, X_test, y_train, y_test = train_test_split(np.arange(0, len(documents)), labels, test_size=0.30, random_state=12)
    x = []
    y = []
    d = []

    save_idx = '/data/sam/' + dataset+ '/train_idx.npy'
    np.save(save_idx, X_train)
    train_distributions = documents[X_train]
    print("Generating dataset......")
    pool = mp.Pool(processes=10)
    jobs = []
    for i in trange(len(train_distributions)):
        for j in range(i + 1, len(train_distributions)):
            d1 = train_distributions[i]
            d2 = train_distributions[j]
            combined_vec = d1 + d2
            non_zero_indices = np.nonzero(combined_vec)[0]
            small_D = D[non_zero_indices][:, non_zero_indices]
            x.append(i)
            y.append(j)
            job = pool.apply_async(ot.emd2, args=(d1[non_zero_indices], d2[non_zero_indices], small_D))
            jobs.append(job)
            #d.append(ot.emd2(d1, d2, D))
    pool.close()
    for job in tqdm(jobs):
        job.wait()
    d = [job.get() for job in jobs]
    dataset = DistanceDataset(torch.tensor(x), torch.tensor(y),  d)
    dataloader = DataLoader(dataset, batch_size=int(len(d)), shuffle=True)
    print("Beginning to construct tree..........")
    tree = OptimizedTree(D)
    print("Starting training............")
    plt_name = dataset + '_losses_50epochs'
    tree.train(dataloader, train_distributions, max_iterations=50, plt_name=plt_name, save_epoch=False)
    save_file = "/data/sam/" + dataset + "/results"
    tree.save_tree_data(save_file, "t1")
    return tree

def average_error(parents, leaves, M_idx, parameters, dataset):
    # lst_names = ["_parents.npy", "_leaves.npy", "_parameters.npy", "_Midx.npy"]
    # data = []
    # for name in lst_names:
    #     f_name = results_path + name
    #     data.append(np.load(f_name))
    word_vecs = np.load('/data/sam/' + dataset + '/vectors.npy')
    word_vecs.astype(np.float32)
    documents = np.load('/data/sam/'+ dataset + '/distributions.npy')
    D = np.load("/data/sam/" + dataset + '/distance_matrix.npy')

    # Get test dataset
    train_indices = np.load("/data/sam/bbc/train_idx.npy")
    test_indices = []
    for i in range(len(documents)):
        if i not in train_indices:
            test_indices.append(i)

    test_documents = documents[test_indices]
    # format test documents
    formatted_test_docs = []
    for distribution in tqdm(test_documents):
        formatted_test_docs.append(format_distributions(distribution))
    print("Number of test documents:", len(formatted_test_docs))

    opt_solver = te.TreeEstimators()
    opt_solver.load_tree(parents.astype(np.int32), leaves.astype(np.int32))

    qt_solver = ote.OTEstimators()
    qt_solver.load_vocabulary(word_vecs)

    opt = np.zeros((len(test_documents), len(test_documents)))
    ft = np.zeros((len(test_documents), len(test_documents)))
    qt = np.zeros((len(test_documents), len(test_documents)))
    ot_d = np.zeros((len(test_documents), len(test_documents)))
    
    opt_time = []
    ft_time = []
    qt_time = []
    sinkhorn_time = []

    pool = mp.Pool(processes=20)
    jobs = []
    for i in trange(len(test_documents)):
        for j in range(i + 1, len(test_documents)):
            d1 = documents[i]
            d2 = documents[j]
            combined_vec = d1 + d2
            non_zero_indices = np.nonzero(combined_vec)[0]
            small_D = D[non_zero_indices][:, non_zero_indices]
            job = pool.apply_async(ot.emd2, args=(d1[non_zero_indices], d2[non_zero_indices], small_D))
            jobs.append(job)
            # ot_d[i][j] = ot.emd2(documents[i], documents[j], D)
            # ot_d[j][i] = ot_d[i][j]
    
    for job in tqdm(jobs):
        job.wait()
    results = [job.get() for job in jobs]
    count = 0
    for i in trange(len(test_documents)):
        for j in range(i + 1, len(test_documents)):
            result = results[count]
            ot_d[i][j] = result
            ot_d[j][i] = result
            count += 1
    
    for i in trange(len(test_documents)):
        for j in range(i + 1, len(test_documents)):
            start = time.time()
            opt[i][j] = tree_wasserstein(M_idx, parameters, opt_solver, formatted_test_docs[i], formatted_test_docs[j])
            opt[j][i] = opt[i][j]
            end = time.time()
            opt_time.append(end - start)
            
            start = time.time()
            ft[i][j] = qt_solver.flowtree_query(formatted_test_docs[i], formatted_test_docs[j])
            ft[j][i] = ft[i][j]
            end = time.time()
            ft_time.append(end - start)

            
            start = time.time()
            qt[i][j] = qt_solver.quadtree_distance(formatted_test_docs[i], formatted_test_docs[j])
            qt[j][i] = qt[i][j]
            end = time.time()
            qt_time.append(end - start)
            
            # ot_d[i][j] = ot.emd2(documents[i], documents[j], D)
            # ot_d[j][i] = ot_d[i][j]
    
    approximations = [opt, ft, qt]
    approx_times = [opt_time, ft_time, qt_time]
    approx_names = ["OptimizedTree", "Flowtree", "Quadtree"]
    for i in range(3):
        print(approx_names[i], "Average relative error:", np.mean(np.abs(approximations[i] - ot_d)), 
              "std dev.:", np.std(np.abs(approximations[i])))
        print("------- Average time per document:", np.mean(approx_times[i]), "std dev:", np.std(approx_times[i]))
    return ot_d, approximations

# takes approximation matrices where approx[i][j] = approx. distance between i and j
# order: optimized tree, flowtree, quadtree
def recall(approximations, real_distance, plt_name):
    names = ["OptimizedTree", "FlowTree", "QuadTree"]
    sorted_distances = np.argsort(real_distance, axis=1)
    top1 = sorted_distances[:, 1]
    all_recall = []
    for approx in approximations:
        approx_sort = np.argsort(approx, axis=1)
        candidates = approx_sort[:, 1:]
        recall = np.zeros(candidates.shape[1])
        for i in range(candidates.shape[0]):
            for j in range(candidates.shape[1]):
                if top1[i] in candidates[i, :j]:
                    recall[j] += 1  
        all_recall.append(recall)  
    for i in range(len(all_recall)):
        recall= all_recall[i]
        plt.plot(np.arange(0, len(recall)), recall, label=names[i])
    plt.legend()
    print("SAVED at", plt_name)
    plt.savefig(plt_name)
    return all_recall

def kNN_experiment(metrics, labels, k = 5):
    for metric in tqdm(metrics):
        metric_sort = np.argsort(metric, axis=1)
        prediction_accuracy = 0
        for i in range(metric_sort.shape[0]):
            top_k = metric_sort[i, 1:k + 1]
            predict = mode(labels[top_k])[0][0]
            if predict == labels[i]:
                prediction_accuracy += 1
        print("Prediction accuracy:", prediction_accuracy)
    

def main():
    # dataset = sys.argv[1]
    # learned_tree = document_classification(dataset)
    # parents = np.load("/data/sam/bbc/results/30/t1/t1_parents.npy")
    # leaves = np.load("/data/sam/bbc/results/30/t1/t1_leaves.npy")
    # parameters = np.load("/data/sam/bbc/results/30/t1/t1_parameters.npy")
    # M_idx = np.load("/data/sam/bbc/results/30/t1/t1_Midx.npy")
    # ot_d, approximations = average_error(parents, leaves, M_idx, parameters, "bbc")
    ot_d =  np.load("/data/sam/bbc/results/30/approximations/OTDistance_t1_test.npy")
    approximations = np.load("/data/sam/bbc/results/30/approximations/approximations_t1.npy")
    #recall(approximations, ot_d, "bbc_recall_t1")
    labels = np.load("/data/sam/bbc/labels.npy")
    kNN_experiment([ot_d], labels, k=1)
    

if __name__=="__main__":
    main()

