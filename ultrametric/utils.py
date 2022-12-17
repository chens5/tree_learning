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
from tqdm import trange
import time

def generate_random_dists(num_dists, n):
    dists = np.random.random_sample((num_dists, n))
    norms = np.linalg.norm(dists, ord=1, axis = 1)
    dists = dists/norms[:, None]
    return dists

def generate_gaussian_dists(num_dists, n):
    dists = np.random.normal(loc=0.0, scale=1.0, size=(num_dists, n))
    norms = np.linalg.norm(dists, ord=1, axis = 1)
    dists = dists/norms[:, None]
    return dists

def generate_random_points(n, dim=2, low=-10, high=10):
    return (high - low) * np.random.rand(n, dim) + low

def generate_gaussian_points(n, dim=2):
    mean = np.zeros(dim)
    cov = np.identity(dim)
    return np.random.multivariate_normal(mean, cov, size=n)

def format_distributions( dist):
    formatted_dist = []
    for i in range(len(dist)):
        if dist[i] > 0:
            formatted_dist.append((i, dist[i]))
    return formatted_dist

