from doctest import OPTIONFLAGS_BY_NAME
import numpy as np
import networkx as nx
import time
import torch
import torchvision
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
import argparse

# A describes distributions i.e. A[0] is a probability distribution over all N words
# Same with B
# returns the tree Wasserstein distances 
def batch_calc_twd(parents, weights, subtree, A, B):
    vA = subtree @ A
    vB = subtree @ B 
    w_v = weights[parents] - weights
    return 0.5  * w_v @ np.abs(vA - vB)
import itertools

def make_subtree(parents, leaves):
    subtree = np.zeros((len(parents), len(leaves)))
    subtree[leaves, leaves] = 1
    iterations = 0
    for leaf in tqdm(leaves):
        cur_node = parents[leaf]
        while cur_node != -1:
            subtree[cur_node][leaf] = 1
            cur_node = parents[cur_node]
            iterations += 1

    return subtree

def average_error(parents, parameters, subtree, word_vecs, documents, distance_mat, normalize=False):
    D = None
    if normalize == True:
        D = distance_mat/distance_mat.max()
        word_vecs = word_vecs/D.max()
    else:
        D = distance_mat

    test_documents = documents
    # format test documents
    formatted_test_docs = []
    for distribution in tqdm(test_documents):
        formatted_test_docs.append(format_distributions(distribution))
    print("Number of test documents:", len(formatted_test_docs))

    qt_solver = ote.OTEstimators()
    qt_solver.load_vocabulary(word_vecs)

    opt_tree = np.zeros((len(test_documents), len(test_documents)))
    ft = np.zeros((len(test_documents), len(test_documents)))
    qt = np.zeros((len(test_documents), len(test_documents)))
    ot_d = np.zeros((len(test_documents), len(test_documents)))
    
    ft_time = []
    qt_time = []

    print("Computing optimized tree wasserstein distances")
    combos = itertools.combinations(np.arange(len(test_documents)), 2)
    l = list(combos)
    indices = [list(t) for t in zip(*l)]
    A = test_documents[indices[0]].transpose()
    B = test_documents[indices[1]].transpose()
    print(".... beginning batch computations ....")
    start = time.time()
    batches_numbers = len(indices[0])//200
    batch_results = []
    for i in trange(batches_numbers):
        if i == batches_numbers-1:
            a = A[:, i * 200: ]
            b = B[:, i * 200: ]
            approx = batch_calc_twd(parents, parameters, subtree, a, b)
            batch_results.append(approx)
        else:
            a = A[:, i*200: (i + 1)*200]
            b = A[:, i*200: (i + 1)*200]
            approx= batch_calc_twd(parents, parameters, subtree, a, b)
            batch_results.append(approx)
    #opt_tree_results = batch_calc_twd(parents, parameters, subtree, A, B)
    end = time.time()
    opt_tree_time = end - start
    opt_tree_results = np.concatenate(batch_results)
    print(opt_tree_results)
    count = 0
    for i in trange(len(test_documents)):
        for j in range(i + 1, len(test_documents)):
            opt_tree[i][j] = opt_tree_results[count]
            opt_tree[j][i] = opt_tree_results[count]
            count += 1

    print("Computing optimal transport distances")
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

    print("Starting flowtree and quadtree computations")
    for i in trange(len(test_documents)):
        for j in range(i + 1, len(test_documents)):

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

    approximations = [ ft, qt]
    approx_times = [ft_time, qt_time]
    approx_names = ["Flowtree", "Quadtree"]
    #print("sinkhorn:", np.mean(sinkhorn_time))
    print("Optimal tree mean absolute error", np.mean(np.abs(opt_tree - ot_d)), "std. dev.:", np.std(np.abs(opt_tree - ot_d)))
    print("------ Average time per document:", opt_tree_time/len(indices[0]))
    for i in range(2):
        print(approx_names[i], "Mean absolute error:", np.mean(np.abs(approximations[i] - ot_d)), 
              "std dev.:", np.std(np.abs(approximations[i])))
        print("------- Average time per document:", np.mean(approx_times[i]), "std dev:", np.std(approx_times[i]))
    return ot_d, ft, qt, opt_tree

def svm(metric, labels, train_indices, t=1):
    test_indices = []
    for i in range(len(labels)):
        if i not in train_indices:
            test_indices.append(i)
    D_train, D_test = metric[train_indices][:, train_indices], metric[test_indices][:, train_indices]
    y_train, y_test = labels[train_indices], labels[test_indices]
    
    svc = SVC(kernel = 'precomputed')
    
    kernel_train = np.exp(-t * D_train)
    
    svc.fit(kernel_train, y_train)
    
    kernel_test = np.exp(-t * D_test)
    
    y_pred = svc.predict(kernel_test)
    return accuracy_score(y_test, y_pred)

def mean_relative_error(original, approximation):
    n = original.shape[0]
    errors = []
    for i in range(n):
        for j in range(i + 1, n):
            if original[i][j] > 0:
                errors.append(abs(original[i][j] - approximation[i][j])/ original[i][j])
                print(approximation[1])
                h()
    return np.mean(errors), np.std(errors) 

def get_remaining_experiments():
    parser = argparse.ArgumentParser(description="Mean relative error, SVM, and nearest neighbor")
    parser.add_argument("--flowtree", type=str)
    parser.add_argument("--quadtree", type=str)
    parser.add_argument("--opttree", type=str)
    parser.add_argument("--ot", type=str)
    parser.add_argument("--SVM", type=bool, default=False)
    parser.add_argument("--knn", type=int, default=0)

    args = parser.parse_args()

    flowtree = np.load(args.flowtree)
    quadtree = np.load(args.quadtree)
    opttree = np.load(args.opttree)
    optimal_transport = np.load(args.ot)

    approximations = [("optimized tree", opttree), ("flowtree", flowtree), ("quadtree", quadtree)]
    for approximation in approximations:
        avg, std = mean_relative_error(optimal_transport, approximation[1])
        print("Approximation:", approximation[0], "--mean relative error:", avg, "standard deviation:", std)
    
    return 



def get_approximations():
    parser = argparse.ArgumentParser(description="Process training parameters")
    parser.add_argument("--dataset_name", type=str, help="Dataset for training")
    parser.add_argument("--trial", type=str, help="trial id")
    parser.add_argument("--distributions", type=str, help="Word distributions")
    parser.add_argument("--word_vectors", type=str, help="word vectors to use")
    parser.add_argument("--labels", type=str, help="labels for distributions")
    parser.add_argument("--train_indices", type=str, help="Saved training indices")
    parser.add_argument("--normalized", type=bool, help="normalize dataset", default=False)

    args = parser.parse_args()

    model_name = '/data/sam/' + args.dataset_name + '/results/' + args.trial + '.pt'
    model_info = torch.load(model_name, map_location='cpu')
    # load model
    parents = model_info['parents']
    leaves = model_info['leaves']
    try:
        subtree = model_info['subtree'].numpy()
    except KeyError:
        subtree = make_subtree(parents, leaves)
    weights = model_info['optimized_state_dict']['param'].numpy()
    # Load data, i.e. distributions, word vectors, and labels 
    print("Dataset:", args.dataset_name)
    word_vecs = np.load(args.word_vectors)
    print("Number of words in dataset", len(word_vecs)) 
    documents =np.load(args.distributions)
    print("Number of documents:", len(documents))
    labels = np.load(args.labels)
    #train_indices = np.load(args.train_indices)
    print("Size of training dataset:", )
    distance_mat = np.load("/data/sam/" + args.dataset_name + '/distance_matrix.npy')


    ot_d, flowtree, quadtree, opttree = average_error(parents, weights, subtree, word_vecs, documents, distance_mat, normalize=args.normalized)

    ot_name = '/data/sam/' + args.dataset_name +'/results/ot_approximations.npy'
    np.save(ot_name, ot_d)
    ft_name = '/data/sam/' + args.dataset_name + '/results/ft_approximations.npy'
    np.save(ft_name, flowtree)
    qt_name = '/data/sam/' + args.dataset_name + '/results/qt_approximations.npy'
    np.save(qt_name, quadtree)
    opttree_name = '/data/sam/' + args.dataset_name + '/results/' + str(args.trial) + 'approximations.npy'
    np.save(opttree_name, opttree)
    

def test_subtree():
    parents = [4, 4, 5, 5, 6, 6, -1]
    leaves = [0, 1, 2, 3]
    print(make_subtree(parents, leaves))

if __name__=="__main__":
    get_approximations()
    #get_remaining_experiments()
    #test_subtree()

