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


def image_classification(trial=0):
    mnist_data = torchvision.datasets.MNIST('/data/sam/mnist', train=True, download=True)
    data = mnist_data.data.numpy()
    labels = mnist_data.targets.numpy()
    pointset = []
    for i in range(28):
        for j in range(28):
            pointset.append([i, j])
    
    D = generate_distance_metric(pointset)
    distributions = []
    for i in range(len(data)):
        d = np.ravel(data[i])
        distributions.append(d/np.linalg.norm(d, ord=1))

    # distributions = np.array(distributions)
    # np.save('/data/sam/mnist/distributions.npy', distributions)
    # np.save('/data/sam/mnist/labels.npy', labels)
    distributions = np.load('/data/sam/mnist/distributions.npy')
    x = []
    y = []
    d = []
    jobs = []
    for i in trange(600):
        idx = np.random.randint(0, len(distributions))
        idy = np.random.randint(0, len(distributions))
        d1 = distributions[idx]
        d2 = distributions[idy]
        combined_vec = d1 + d2
        non_zero_indices = np.nonzero(combined_vec)[0]
        small_D = D[non_zero_indices][:, non_zero_indices]
        x.append(d1)
        y.append(d2)
        d.append(ot.emd2(d1[non_zero_indices], d2[non_zero_indices], small_D))

    # for i in trange(len(train_distributions)):
    #     for j in range(i + 1, len(train_distributions)):
    #         d1 = train_distributions[i]
    #         d2 = train_distributions[j]
    #         combined_vec = d1 + d2
    #         non_zero_indices = np.nonzero(combined_vec)[0]
    #         #D = ot.dist(word_vecs[non_zero_indices], metric='euclidean')
    #         small_D = D[non_zero_indices][:, non_zero_indices]
    #         x.append(i)
    #         y.append(j)
    #         job = pool.apply_async(ot.emd2, args=(d1[non_zero_indices], d2[non_zero_indices], small_D))
    #         jobs.append(job)
    #             #d.append(ot.emd2(d1, d2, D))
    # pool.close()
    # for job in tqdm(jobs):
    #     job.wait()
    # d = [job.get() for job in jobs]
    dataset = DistanceDataset(torch.tensor(x), torch.tensor(y),  d)
    dataloader = DataLoader(dataset, batch_size=200, shuffle=True)
    print("Beginning to construct tree..........")
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    tree = GPUOptimizedTree(D, device=device)
    tree.to(device)

    optimizer = torch.optim.Adam(tree.parameters(), lr = 0.01)

    print("Starting training............")
    plt_name ='mnist_losses_50epochs_t' + str(trial)
    tree.train(dataloader, optimizer, distributions, max_iterations=300, plt_name=plt_name, save_epoch=False)
    #save_file = "/data/sam/" + dataset_name + "/results"
    store_tree('mnist', trial, tree)
    return tree

def hinge_loss_training(dataset_name, word_vecs, documents, labels, trial=0, sample=False, device='cuda:1'):
    print("Generating distance matrix.....")
    D = None
    distance_matrix_name = '/data/sam/' + dataset_name + '/distance_matrix.npy'
    try:
        D = np.load(distance_matrix_name)
    except OSError:
        D=generate_distance_metric(word_vecs)
        np.save(distance_matrix_name, D)
    D = D
    word_vecs = word_vecs
    rs = 25
    test_sz = 30

    # split into test/train
    X_train, X_test, y_train, y_test = train_test_split(np.arange(0, len(documents)), labels, test_size=test_sz/100, random_state=rs)
    x = []
    y = []
    d = []

    # # save test/train split
    save_idx = '/data/sam/' + dataset_name+ '/test_idx_test' + str(test_sz) + '_rs' + str(rs)+ '.npy'
    np.save(save_idx, X_test)

    # # save test labels
    save_test_labels = '/data/sam/' + dataset_name + '/test_labels_test' + str(test_sz) + '_rs' + str(rs)+ '.npy'
    np.save(save_test_labels, y_test)

    train_distributions = documents[X_train]
    #train_distributions = documents
    
    print("Number of training distributions", train_distributions.shape[0])
    print("Generating dataset......")

    # Loading training datasets 
    if sample:
        print("Not sampling pairs")
        d = []
        for i in range(len(train_distributions)):
            for j in range(i + 1, len(train_distributions)):
                x.append(i)
                y.append(j)
                if y_train[i] == y_train[j]:
                    d.append(1)
                else:
                    d.append(0)
    else:
        d = []
        xvals = np.random.choice(len(train_distributions), 1000000)
        yvals = np.random.choice(len(train_distributions), 1000000)
        x = xvals
        y = yvals
        save_xval_name = '/data/sam/' + dataset_name + '/trainx.npy'
        save_yval_name = '/data/sam/' + dataset_name + '/trainy.npy'
        np.save(save_xval_name, xvals)
        np.save(save_yval_name, yvals)
        d = []
        for i in xvals:
            for j in yvals:
                if y_train[i] == y_train[j]:
                    d.append(1)
                else:
                    d.append(0)

    print("number of pairs", len(x))
    dataset = DistanceDataset(torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long),  torch.tensor(d))
    dataloader = DataLoader(dataset, batch_size=3000, shuffle=True)
    print("Beginning to construct tree..........")
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    tree = GPUOptimizedTree(D, device=device)
    tree.to(device)

    optimizer = torch.optim.Adam(tree.parameters(), lr = 0.1)

    print("Starting training............")
    plt_name =dataset_name + '_losses_50epochs_t' + str(trial)
    tree.train(dataloader, optimizer, train_distributions, max_iterations=100, plt_name=plt_name, save_epoch=False, contrastive=True)
    save_file = "/data/sam/" + dataset_name + "/results/" + str(trial) + ".pt"
    torch.save({'parents': tree.parents,
                'leaves': tree.leaves,
                'subtree':tree.subtree,
                'optimized_state_dict': tree.state_dict()}, 
                save_file)
    return tree

def document_classification(dataset_name, word_vecs, documents, labels, trial=0, sample=False, device="cuda:1"):
    print("Generating distance matrix.....")
    D = None
    distance_matrix_name = '/data/sam/' + dataset_name + '/distance_matrix.npy'
    try:
        D = np.load(distance_matrix_name)
    except OSError:
        D=generate_distance_metric(word_vecs)
        np.save(distance_matrix_name, D)
    D = D
    word_vecs = word_vecs
    rs = 30
    test_sz = 30

    # split into test/train
    X_train, X_test, y_train, y_test = train_test_split(np.arange(0, len(documents)), labels, test_size=test_sz/100, random_state=rs)
    x = []
    y = []
    d = []

    # # save test/train split
    save_idx = '/data/sam/' + dataset_name+ '/test_idx_test' + str(test_sz) + '_rs' + str(rs)+ '.npy'
    np.save(save_idx, X_test)

    # # save test labels
    save_test_labels = '/data/sam/' + dataset_name + '/test_labels_test' + str(test_sz) + '_rs' + str(rs)+ '.npy'
    np.save(save_test_labels, y_test)

    train_distributions = documents[X_train]
    #train_distributions = documents
    
    print("Number of training distributions", train_distributions.shape[0])
    print("Generating dataset......")

    # Loading training datasets 
    try:
        save_dataset = '/data/sam/' + dataset_name + '/dataset_test' + str(test_sz) + '_rs' + str(rs)+ '.npy'
        d = np.load(save_dataset)
        for i in range(len(train_distributions)):
            for j in range(i + 1, len(train_distributions)):
                x.append(i)
                y.append(j)
    
    except OSError:
        pool = mp.Pool(processes=20)
        jobs = []
        if sample==True:
            xvals = np.random.choice(len(train_distributions), 1000000)
            yvals = np.random.choice(len(train_distributions), 1000000)
            save_xval_name = '/data/sam/' + dataset_name + '/trainx.npy'
            save_yval_name = '/data/sam/' + dataset_name + '/trainy.npy'
            np.save(save_xval_name, xvals)
            np.save(save_yval_name, yvals)
            for i in trange(len(xvals)):
                d1 = train_distributions[xvals[i]]
                d2 = train_distributions[yvals[i]]
                combined_vec = d1 + d2
                non_zero_indices = np.nonzero(combined_vec)[0]
                #D = ot.dist(word_vecs[non_zero_indices], metric='euclidean')
                small_D = D[non_zero_indices][:, non_zero_indices]
                x.append(xvals[i])
                y.append(yvals[i])
                job = pool.apply_async(ot.emd2, args=(d1[non_zero_indices], d2[non_zero_indices], small_D))
                jobs.append(job)
        else:
            for i in trange(len(train_distributions)):
                for j in range(i + 1, len(train_distributions)):
                    d1 = train_distributions[i]
                    d2 = train_distributions[j]
                    combined_vec = d1 + d2
                    non_zero_indices = np.nonzero(combined_vec)[0]
                    #D = ot.dist(word_vecs[non_zero_indices], metric='euclidean')
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
        save_dataset = '/data/sam/' + dataset_name + '/dataset_test' + str(test_sz) + '_rs' + str(rs)+ '.npy'
        np.save(save_dataset, d)
    print("number of pairs", len(x))
    dataset = DistanceDataset(torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long),  torch.tensor(d))
    dataloader = DataLoader(dataset, batch_size=2000, shuffle=True)
    print("Beginning to construct tree..........")
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    tree = GPUOptimizedTree(D, device=device)
    tree.to(device)

    optimizer = torch.optim.Adam(tree.parameters(), lr = 0.1, )

    print("Starting training............")
    plt_name =dataset_name + '_losses_50epochs_t' + str(trial)
    tree.train(dataloader, optimizer, train_distributions, max_iterations=50, plt_name=plt_name, save_epoch=False)
    save_file = "/data/sam/" + dataset_name + "/results/" + str(trial) + ".pt"
    torch.save({'parents': tree.parents,
                'leaves': tree.leaves,
                'subtree':tree.subtree,
                'optimized_state_dict': tree.state_dict()}, 
                save_file)
    return tree

def main():
    #image_classification()
    parser = argparse.ArgumentParser(description="Process training parameters")
    parser.add_argument("--dataset_name", type=str, help="Dataset for training")
    parser.add_argument("--trial", type=int, help="trial id")
    parser.add_argument("--is_sampling", type=bool, help="Sample pairs from training dataset.", default=False)
    parser.add_argument("--device", type=str, help="GPU")
    parser.add_argument("--distributions", type=str, help="Word distributions")
    parser.add_argument("--word_vectors", type=str, help="word vectors to use")
    parser.add_argument("--labels", type=str, help="labels for distributions")
    parser.add_argument("--contrastive", type=bool, default=False)
    #parser.add_argument("--save_file", type=str, help="save file name")
    args = parser.parse_args()
    # f = open("/data/sam/twitter/results/"+str(args.trial) + "specifications.txt", 'w')
    # f.write(args)

    # Load dataset
    print("Dataset:", args.dataset_name)
    word_vecs = np.load(args.word_vectors)
    print("Number of words in dataset", len(word_vecs)) 
    documents =np.load(args.distributions)
    print("Number of documents:", len(documents))
    labels = np.load(args.labels)
    
    print("Starting training for", args.dataset_name)
    print("Sampling pairs:", args.is_sampling)
    if args.contrastive:
        hinge_loss_training(args.dataset_name, word_vecs, documents, labels, trial=args.trial, sample=args.is_sampling, device=args.device)
    else:
        document_classification(args.dataset_name, word_vecs, documents, labels, trial=args.trial, sample = args.is_sampling, device=args.device)
    #load_tree(dataset_name, 1)


    
if __name__=="__main__":
    main()