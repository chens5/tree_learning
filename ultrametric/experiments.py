from calendar import c
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import sys
from scipy.stats import mode
import argparse
import scipy.io
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning



# A describes distributions i.e. A[0] is a probability distribution over all N words
# Same with B
# returns the tree Wasserstein distances 
def batch_calc_twd(parents, weights, subtree, A, B):
    vA = subtree @ A
    vB = subtree @ B 
    w_v = weights[parents] - weights
    return 0.5  * w_v @ np.abs(vA - vB)

def batch_calc_qt(parents, weights, subtree, A, B):
    vA = subtree @ A
    vB = subtree @ B 
    w_v = weights[parents] - weights
    return w_v @ np.abs(vA - vB)

def make_subtree(parents, leaves):
    n = len(leaves)
    subtree = np.zeros((len(parents), n))
    for i in range(n):
        leaf_index = i 
        leaf_node = leaves[i]
        parent_node = parents[leaf_node]
        subtree[leaf_node][leaf_index] = 1
        while parent_node != -1:
            subtree[parent_node][leaf_index] = 1
            parent_node =parents[parent_node]
    return subtree

def get_weights(parents, delta):
    num_nodes = len(parents)
    weights = np.zeros(num_nodes)
    for i in range(num_nodes):
        node = i
        parent = parents[node]
        lvl = 0
        while parent != -1:
            parent = parents[parent]
            lvl += 1
        weights[i] = delta/(2**lvl)
    return weights


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
    leaves_qt = qt_solver.return_leaf_details()
    parents_qt = qt_solver.return_parent_details()
    delta_qt = qt_solver.return_bounding_box()
    subtree_qt = make_subtree(parents_qt, leaves_qt)
    weights_qt = get_weights(parents_qt, delta_qt)

    
    ft_time = []
    qt_time = []

    print("Computing optimized tree wasserstein distances")
    n_documents = len(test_documents)
    ind1 = []
    ind2 = []
    for i in range(n_documents):
        for j in range(i + 1, n_documents):
            ind1.append(i)
            ind2.append(j)
    n_samples = len(ind1)

    A = test_documents[ind1].transpose()
    B = test_documents[ind2].transpose()
    print(".... beginning batch computations ....")
    start = time.time()
    nbat = n_samples//200
    bres = []
    bres_qt = []
    for ii in trange(nbat):
        if ii < nbat - 1:
            ans = batch_calc_twd(parents, parameters, subtree, A[:, ii*200: (ii + 1)*200], B[:, ii*200 : (ii + 1)*200])
            ans_qt = batch_calc_twd(parents_qt, weights_qt, subtree_qt, A[:, ii*200: (ii + 1)*200], B[:, ii*200 : (ii + 1)*200])
        else:
            ans = batch_calc_twd(parents, parameters, subtree, A[:, ii*200:], B[:, ii*200:])
            ans_qt = batch_calc_twd(parents_qt, weights_qt, subtree_qt, A[:, ii*200:], B[:, ii*200 :])
        bres.append(ans)  
        bres_qt.append(ans_qt)

    ot_approximation = np.concatenate(bres)
    qt=np.concatenate(bres_qt)
    end = time.time()
    opt_tree_time = end - start
    opt_tree_results = ot_approximation

    print("Computing optimal transport distances")
    pool = mp.Pool(processes=20)
    jobs = []
    for ii in trange(n_samples):
        d1 = test_documents[ind1[ii]]
        d2 = test_documents[ind2[ii]]
        combined_vec = d1 + d2
        non_zero_indices = np.nonzero(combined_vec)[0]
        small_D = D[non_zero_indices][:, non_zero_indices]
        job = pool.apply_async(ot.emd2, args=(d1[non_zero_indices], d2[non_zero_indices], small_D))
        jobs.append(job)
    
    for job in tqdm(jobs):
        job.wait()
    ground_truth = [job.get() for job in jobs]
    # ground_truth = np.load('/data/sam/twitter/results/ot_approximations.npy')
    # print(np.mean(np.abs(ground_truth - qt)), np.std(np.abs(ground_truth - qt)))
    # np.save('/data/sam/twitter/results/qt_approximations.npy', qt)
    # return

    print("Starting flowtree and quadtree computations")
    ft = []
    qt = []

    for ii in trange(n_samples):
        a = formatted_test_docs[ind1[ii]]
        b = formatted_test_docs[ind2[ii]]
        start = time.time()
        ft.append(qt_solver.flowtree_query(a, b))
        end = time.time()
        ft_time.append(end - start)
        
        a = formatted_test_docs[ind1[ii]]
        b = formatted_test_docs[ind2[ii]]
        start = time.time()
        qt.append(qt_solver.quadtree_distance(a, b))
        end = time.time()
        qt_time.append(end - start)
        
    approximations = [ np.array(ft), np.array(qt)]
    approx_times = [ft_time, qt_time]
    approx_names = ["Flowtree", "Quadtree"]
    print("Optimal tree mean absolute error", np.mean(np.abs(opt_tree_results- ground_truth)), "std. dev.:", np.std(np.abs(opt_tree_results - ground_truth)))
    print("------ Average time per document:", opt_tree_time/n_samples)
    for i in range(2):
        print(approx_names[i], "Mean absolute error:", np.mean(np.abs(approximations[i] - ground_truth)), 
              "std dev.:", np.std(np.abs(approximations[i])))
        print("------- Average time per document:", np.mean(approx_times[i]), "std dev:", np.std(approx_times[i]))
    return ground_truth, ft, qt, opt_tree_results

def svm(metric, labels, train_indices, t=1, C=1):
    test_indices = []
    for i in range(len(labels)):
        if i not in train_indices:
            test_indices.append(i)
    D_train, D_test = metric[train_indices][:, train_indices], metric[test_indices][:, train_indices]
    y_train, y_test = labels[train_indices], labels[test_indices]
    
    svc = SVC(kernel = 'precomputed', C=1, max_iter=5000)
    
    kernel_train = np.exp(-t * D_train)
    
    svc.fit(kernel_train, y_train)
    
    kernel_test = np.exp(-t * D_test)
    
    y_pred = svc.predict(kernel_test)
    return accuracy_score(y_test, y_pred)

def knn(metric, labels, train_indices, k=1):
    test_indices = []
    for i in range(len(labels)):
        if i not in train_indices:
            test_indices.append(i)
    D_train, D_test = metric[train_indices][:, train_indices], metric[test_indices][:, train_indices]
    y_train, y_test = labels[train_indices], labels[test_indices]
    clf = KNeighborsClassifier(n_neighbors = k, metric='precomputed')
    clf.fit(D_train, y_train)
    y_pred = clf.predict(D_test)
    return accuracy_score(y_test, y_pred)

def mean_relative_error(original, approximation):
    n = original.shape[0]
    errors = []
    for i in range(n):
        for j in range(i + 1, n):
            if original[i][j] > 0:
                errors.append(abs(original[i][j] - approximation[i][j])/ original[i][j])
    return np.mean(errors), np.std(errors) 

def metric_to_distance_matrix(n, metric):
    matrix = np.zeros((n, n))
    count = 0
    for i in range(n):
        for j in range(i+ 1, n):
            matrix[i][j] = metric[count]
            matrix[j][i] = metric[count]
            count += 1
    return matrix

@ignore_warnings(category=ConvergenceWarning)
def choose_parameters(D, param_list, labels):
    skf = StratifiedKFold(n_splits = 5, shuffle=True)
    results = []
    for train_index, test_index in skf.split(D, labels):
        D_train, D_test = D[train_index][:, train_index], D[test_index][:, train_index]
        y_train, y_test = labels[train_index], labels[test_index]
        split_results = []
        for pair in param_list:
            t = pair[0]
            C = pair[1]
            K_train = np.exp(-t * D_train)
            K_test = np.exp(-t * D_test)
            clf = SVC(kernel='precomputed', C = C)
            clf.fit(K_train, y_train)
            y_pred = clf.predict(K_test)
            split_results.append(accuracy_score(y_test, y_pred))
        results.append(split_results)
    results = np.array(results)
    fin_results = results.mean(axis=0)
    best_idx = np.argmax(fin_results)
    print(fin_results[best_idx])
    return param_list[best_idx]

def get_remaining_experiments():
    parser = argparse.ArgumentParser(description="Mean relative error, SVM, and nearest neighbor")
    parser.add_argument("--flowtree", type=str)
    parser.add_argument("--quadtree", type=str)
    parser.add_argument("--opttree", type=str)
    parser.add_argument("--ot", type=str)
    parser.add_argument("--SVM", type=bool, default=False)
    parser.add_argument("--knn", type=int, default=0)
    parser.add_argument("--matfile", type=str, default=0)

    args = parser.parse_args()

    flowtree = np.load(args.flowtree)
    quadtree = np.load(args.quadtree)
    opttree = np.load(args.opttree)
    optimal_transport = np.load(args.ot)
    print("shapes", flowtree.shape, quadtree.shape, opttree.shape, optimal_transport.shape)

    # load labels
    f = scipy.io.loadmat(args.matfile)
    TR = np.array(f['TR'])
    labels = np.array(f['Y'])[0]
    ts = [0.01, 0.1, 1, 10, 100]
    Cs = [0.01, 0.1, 1, 10, 100]
    param_pairs = []
    n_docs = labels.shape[0]
    metrics = [flowtree, quadtree, opttree, optimal_transport]
    distance_matrices = []

    for i in range(4):
        distance_matrices.append(metric_to_distance_matrix(n_docs, metrics[i]))

    for i in ts:
        for j in Cs:
            param_pairs.append((i, j))

    print("----- Starting SVM classification")
    if args.SVM:
        name = ["flowtree", "quadtree", "opttree", "optimal transport"]
        for k in range(4):
            metric = distance_matrices[k]
            results = []
            for i in range(5):
                TR = np.array(f['TR'])[i] - 1
                param_res = []
                for param in param_pairs:
                    param_res.append(svm(metric, labels,TR,t=param[0], C=param[1]))
                idx = np.argmax(param_res)
                print("Best parameters:", param_pairs[idx])
                results.append(param_res[idx])
            print(name[k], "Average accuracy", np.mean(results), "Standard deviation", np.std(results))
    
    print("--------- Starting K-Nearest Neighbors classifications")
    if args.knn > 0:
        name = ["flowtree", "quadtree", "opttree", "optimal transport"]
        for k in range(4):
            metric = distance_matrices[k]
            results = []
            for i in range(5):
                TR = np.array(f['TR'])[i] - 1
                results.append(knn(metric, labels,TR,k=args.knn))
            print(name[k], "Average accuracy", np.max(results), "Standard deviation", np.std(results))

    



def get_approximations():
    parser = argparse.ArgumentParser(description="Process training parameters")
    parser.add_argument("--dataset_name", type=str, help="Dataset for training")
    parser.add_argument("--trial", type=str, help="trial id")
    parser.add_argument("--distributions", type=str, help="Word distributions")
    parser.add_argument("--word_vectors", type=str, help="word vectors to use")
    parser.add_argument("--labels", type=str, help="labels for distributions")
    parser.add_argument("--train_indices", type=str, help="Saved training indices")
    parser.add_argument("--normalized", type=bool, help="normalize dataset", default=False)
    parser.add_argument("--npz", type=str)

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

    # Load dataset
    if args.npz:
        save_file = np.load(args.npz, allow_pickle=True)
        word_vecs = np.float32(save_file['vectors'])
        documents = save_file['distributions']

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
    parents = [-1, 5, 5, 6, 6, 0, 0]
    leaves = [1, 2, 3, 4]
    subtree = make_subtree(parents, leaves)
    weights = get_weights(parents, 32)
    a = np.array([0.5, 0.5, 0, 0])
    b = np.array([0, 0, 0.5, 0.5])
    print(batch_calc_qt(parents, weights, subtree, a, b))

if __name__=="__main__":
    #get_approximations()
    get_remaining_experiments()
