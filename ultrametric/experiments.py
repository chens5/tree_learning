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


def document_classification():
    word_vecs = np.load('/data/sam/twitter/vectors.npy')
    documents = np.load('/data/sam/twitter/distributions.npy')
    labels = np.load('/data/sam/twitter/labels.npy')
    #print("Generating distance matrix.....")
    #D = generate_distance_metric(word_vecs)
    #np.save("distance_matrix_twitter.npy", D)
    D = np.load("distance_matrix_twitter.npy")

    X_train, X_test, y_train, y_test = train_test_split(np.arange(0, len(documents)), labels, test_size=0.66, random_state=42)
    x = []
    y = []
    d = []
    np.save("/data/sam/twitter/train_idx1.npy", X_train)
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
    dataloader = DataLoader(dataset, batch_size=int(len(d)/2), shuffle=True)
    print("Beginning to construct tree..........")
    tree = OptimizedTree(D)
    print("Starting training............")
    tree.train(dataloader, train_distributions, max_iterations=50, plt_name="twitter_losses_30epoch", save_epoch=True)
    tree.save_tree_data("/data/sam/twitter/results/30", "t2")
  
def main():
    document_classification()
        
    
if __name__=="__main__":
    main()


