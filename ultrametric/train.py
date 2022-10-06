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
import scipy.io
from torch.optim.lr_scheduler import ExponentialLR

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


def image_classification(dataset_name, trial=0, device='cuda:1'):
    foldername = '/data/sam/' + dataset_name
    mnist_data = torchvision.datasets.MNIST(foldername, train=True, download=True)
    data = mnist_data.data.numpy()
    pointset = []
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            pointset.append([i, j])
    pointset = np.array(pointset)
    #D = ot.dist(pointset, pointset, metric='minkowski', p=1)
    D = ot.dist(pointset, pointset, metric='euclidean')
    distributions = []
    for i in range(len(data)):
        d = np.ravel(data[i])
        distributions.append(d/np.linalg.norm(d, ord=1))

    distributions = np.array(distributions)
    num_dists = len(distributions)
    X_train, X_test, y_train, y_test = train_test_split(np.arange(0, num_dists), np.zeros(num_dists), test_size=20)
    dist_sf = '/data/sam/' + dataset_name + '/trainx.npy'
    np.save(dist_sf, X_train)
    # np.save('/data/sam/mnist/labels.npy', labels)
    # distributions = np.load('/data/sam/mnist/distributions.npy')
    train_distributions = distributions[X_train]
    x = []
    y = []
    d = []
    pool = mp.Pool(processes=20)
    jobs = []
    for i in trange(1000000):
        idx = np.random.randint(0, len(train_distributions))
        idy = np.random.randint(0, len(train_distributions))
        d1 = train_distributions[idx]
        d2 = train_distributions[idy]
        combined_vec = d1 + d2
        non_zero_indices = np.nonzero(combined_vec)[0]
        small_D = D[non_zero_indices][:, non_zero_indices]
        x.append(idx)
        y.append(idy)
        job = pool.apply_async(ot.emd2, args=(d1[non_zero_indices], d2[non_zero_indices], small_D))
        jobs.append(job)
    for job in tqdm(jobs):
        job.wait()
    d = [job.get() for job in jobs]
    
    dataset = DistanceDataset(torch.tensor(x), torch.tensor(y),  torch.tensor(d))
    dataloader = DataLoader(dataset, batch_size=2048, shuffle=True)
    print("Beginning to construct tree..........")
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    tree = GPUOptimizedTree(D, device=device)
    tree.to(device)

    optimizer = torch.optim.Adam(tree.parameters(), lr = 0.1)

    print("Starting training............")
    plt_name =dataset_name + str(trial)
    fname = "/data/sam/" + dataset_name + "/results/"
    tree.train(dataloader, optimizer, train_distributions, max_iterations=100, plt_name=plt_name, save_epoch=True, bsz=4, filename=fname)
    save_file = "/data/sam/" + dataset_name + "/results/" + str(trial) + '.pt' 
    #store_tree('mnist', trial, tree)
    torch.save({'parents': tree.parents,
                'leaves': tree.leaves,
                'subtree':tree.subtree,
                'vec': tree.vec,
                'M_idx': tree.M_idx,
                'np_parameters': tree.np_parameters,
                'optimized_state_dict': tree.state_dict()}, 
                save_file)
    return tree

def weight_optimization(dataset_name, word_vecs, documents, TR=None, trial=0, sample=False, device='cuda:1', quadtree=False, bsz=-1):
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
    rs = 314
    test_sz = 20

    # split into test/train
    if len(TR) != 0:
        X_train = TR
    else:
        X_train, X_test, y_train, y_test = train_test_split(np.arange(0, len(documents)), np.zeros(len(documents)), test_size=test_sz/100, random_state=rs)
    x = []
    y = []
    d = []

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
            
            xvals = np.random.choice(len(train_distributions), 100000)
            yvals = np.random.choice(len(train_distributions), 100000)
            m = len(train_distributions)
            if 1000000 < (m * (m - 1)/2):
                xvals = []
                yvals = []
                for i in range(m):
                    for j in range(m):
                        xvals.append(i)
                        yvals.append(j)
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

    # to do quadtree weight training, we must generate the subtree matrix

    optimizer = torch.optim.Adam(tree.parameters(), lr = 0.1, )

    print("Starting training............")
    plt_name =dataset_name + '_losses_50epochs_t' + str(trial)
    save_file = "/data/sam/" + dataset_name + "/results/" + str(trial) 
    tree.train_weights(dataloader, optimizer, train_distributions, max_iterations=50, plt_name=plt_name, save_epoch=True, bsz=bsz, filename=save_file)
    save_file = "/data/sam/" + dataset_name + "/results/" + str(trial) + ".pt"
    torch.save({'parents': tree.parents,
                'leaves': tree.leaves,
                'subtree':tree.subtree,
                'vec': tree.vec,
                'M_idx': tree.M_idx,
                'np_parameters': tree.np_parameters,
                'optimized_state_dict': tree.state_dict()}, 
                save_file)
    return tree

def hinge_loss_training(dataset_name, word_vecs, documents, TR=None, trial=0, sample=False, device='cuda:1'):
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
    if TR:
        X_train = TR
    else:
        X_train, X_test, y_train, y_test = train_test_split(np.arange(0, len(documents)), np.zeros(len(documents)), test_size=test_sz/100, random_state=rs)
    x = []
    y = []
    d = []


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

def document_classification(dataset_name, word_vecs, documents, TR=[], trial=0, sample=False, device="cuda:1", bsz=-1):
    print("Generating distance matrix.....")
    D = None
    distance_matrix_name = '/data/sam/' + dataset_name + '/distance_matrix.npy'
    try:
        D = np.load(distance_matrix_name)
    except OSError:
        #D=generate_distance_metric(word_vecs)
        D = ot.dist(word_vecs, word_vecs, metric='euclidean')
        np.save(distance_matrix_name, D)
    D = D
    word_vecs = word_vecs
    rs = 320
    test_sz = 20

    # split into test/train
    if len(TR) != 0:
        X_train = TR
    else:
        X_train, X_test, y_train, y_test = train_test_split(np.arange(0, len(documents)), np.zeros(len(documents)), test_size=test_sz/100, random_state=rs)
    x = []
    y = []
    d = []

    train_distributions = documents[X_train]
    #train_distributions = documents
    
    print("Number of training distributions", train_distributions.shape[0])
    print("Generating dataset......")

    # Loading training datasets 
    try:
        save_dataset = '/data/sam/' + dataset_name + '/dataset_test' + str(test_sz) + '_rs' + str(rs)+ '.npy'
        save_xval_name = '/data/sam/' + dataset_name + '/trainx.npy'
        save_yval_name = '/data/sam/' + dataset_name + '/trainy.npy'
        d = np.load(save_dataset)
        x = np.load(save_xval_name)
        y = np.load(save_yval_name)
    
    except OSError:
        pool = mp.Pool(processes=20)
        jobs = []
        if sample==True:
            
            xvals = np.random.choice(len(train_distributions), 1000000)
            yvals = np.random.choice(len(train_distributions), 1000000)
            m = len(train_distributions)
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

    optimizer = torch.optim.Adam(tree.parameters(), lr = 0.1 )

    print("Starting training............")
    plt_name =dataset_name + '_losses_50epochs_t' + str(trial)
    save_file = "/data/sam/" + dataset_name + "/results/" + str(trial) 
    tree.train(dataloader, optimizer, train_distributions, max_iterations=50, plt_name=plt_name, save_epoch=True, bsz=-1, filename=save_file)
    save_file = "/data/sam/" + dataset_name + "/results/" + str(trial) + ".pt"
    torch.save({'parents': tree.parents,
                'leaves': tree.leaves,
                'subtree':tree.subtree,
                'vec': tree.vec,
                'M_idx': tree.M_idx,
                'np_parameters': tree.np_parameters,
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
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--npz", type=str)
    parser.add_argument('--matfile', type=str)
    parser.add_argument("--image_dataset", type=str)
    #parser.add_argument("--save_file", type=str, help="save file name")
    args = parser.parse_args()
    # f = open("/data/sam/twitter/results/"+str(args.trial) + "specifications.txt", 'w')
    # f.write(args)

    if args.image_dataset != None:
        image_classification(args.image_dataset, trial=args.trial, device=args.device)
        return

    # Load dataset
    print("Loading dataset")
    if args.npz:
        save_file = np.load(args.npz, allow_pickle=True)
        word_vecs = save_file['vectors']
        documents = save_file['distributions']
        f = scipy.io.loadmat(args.matfile)
        TR = np.array(f['TR'])[0] - 1
    else:
        print("Dataset:", args.dataset_name)
        word_vecs = np.load(args.word_vectors)
        print("Number of words in dataset", len(word_vecs)) 
        documents =np.load(args.distributions)
        print("Number of documents:", len(documents))
        labels = np.load(args.labels)
        TR = []
    
    print("Starting training for", args.dataset_name)
    print("Sampling pairs:", args.is_sampling)
    if args.contrastive:
        hinge_loss_training(args.dataset_name, word_vecs, documents, labels, trial=args.trial, sample=args.is_sampling, device=args.device)
    else:
        document_classification(args.dataset_name, word_vecs, documents, TR=TR, trial=args.trial, sample = args.is_sampling, device=args.device, bsz=8000)
    #load_tree(dataset_name, 1)

if __name__=="__main__":
    main()