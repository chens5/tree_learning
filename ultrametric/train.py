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

def store_tree(dataset_name, trial, tree):
    save_file_name = '/data/sam/' + dataset_name + '/results/tree' + str(trial) + '.pickle'
    file_to_store = open(save_file_name, "wb")
    pickle.dump(tree, file_to_store)
    file_to_store.close()

def load_tree(dataset_name, trial):
    save_file_name = '/data/sam/' + dataset_name + '/results/tree' + str(trial) + '.pickle'
    file_to_read = open(save_file_name, "rb")
    tree = pickle.load(file_to_read)
    file_to_read.close()
    return tree

def document_classification(dataset_name, trial=0):
    print("Dataset:", dataset_name)
    wv_fname = '/data/sam/' + dataset_name+ '/vectors.npy'
    word_vecs = np.load(wv_fname)
    print("Number of words in dataset", len(word_vecs)) 
    document_fname = '/data/sam/' + dataset_name + '/distributions.npy'
    documents = np.load(document_fname)
    print("Number of documents:", len(documents))
    lbl_fname = '/data/sam/' + dataset_name + '/labels.npy'
    labels = np.load(lbl_fname)
    print("Loaded all data")
    print("Generating distance matrix.....")
    #D = generate_distance_metric(word_vecs)
    distance_mat_name = '/data/sam/' + dataset_name + '/distance_matrix.npy'
    #np.save(distance_mat_name, D)
    D = np.load(distance_mat_name)

    rs = 12

    X_train, X_test, y_train, y_test = train_test_split(np.arange(0, len(documents)), labels, test_size=0.30, random_state=rs)
    x = []
    y = []
    d = []

    save_idx = '/data/sam/' + dataset_name+ '/test_idx.npy'
    np.save(save_idx, X_train)

    save_test_labels = '/data/sam/' + dataset_name + '/test_labels.npy'
    np.save(save_test_labels, y_test)

    train_distributions = documents[X_train]
    print("Generating dataset......")
    try:
        save_dataset = '/data/sam/' + dataset_name + '/dataset_rs' + str(rs) + '.npy'
        d = np.load(save_dataset)
        for i in range(len(train_distributions)):
            for j in range(i + 1, len(train_distributions)):
                x.append(i)
                y.append(j)
    except OSError:
        pool = mp.Pool(processes=20)
        jobs = []
        for i in trange(len(train_distributions)):
            for j in range(i + 1, len(train_distributions)):
                d1 = train_distributions[i]
                d2 = train_distributions[j]
                combined_vec = d1 + d2
                non_zero_indices = np.nonzero(combined_vec)[0]
                # small_D = D[non_zero_indices][:, non_zero_indices]
                # print(len(non_zero_indices))
                # start = time.time()
                # ot.emd2(d1[non_zero_indices], d2[non_zero_indices], small_D)
                # end = time.time()
                # print("TIME:", end - start)
                # h()
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
        save_dataset = '/data/sam/' + dataset_name + '/dataset_rs12.npy'
        np.save(save_dataset, d)

    dataset = DistanceDataset(torch.tensor(x), torch.tensor(y),  d)
    dataloader = DataLoader(dataset, batch_size=2000, shuffle=True)
    print("Beginning to construct tree..........")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tree = GPUOptimizedTree(D, device=device)
    tree.to(device)

    optimizer = torch.optim.Adam(tree.parameters(), lr = 0.1)

    print("Starting training............")
    plt_name =dataset_name + '_losses_50epochs' + str(trial)
    tree.train(dataloader, optimizer, train_distributions, max_iterations=1, plt_name=plt_name, save_epoch=False)
    #save_file = "/data/sam/" + dataset_name + "/results"
    store_tree(dataset_name, trial, tree)

    

    return tree

def main():
    dataset_name = sys.argv[1]
    trial = sys.argv[2]
    print("Starting training for", dataset_name)
    learned_tree = document_classification(dataset_name, trial=trial)
    #load_tree(dataset_name, 1)


    
if __name__=="__main__":
    main()