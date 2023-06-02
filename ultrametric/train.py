import numpy as np
import torch
import ot
from networkx.drawing.nx_pydot import *
from tqdm import trange, tqdm
import time
from utils import *
import multiprocessing as mp
from torch.utils.data import DataLoader
from ot_tree import *
from sklearn.model_selection import train_test_split
import argparse
import scipy.io
from cluster_tree import *
import os

WORD_DATASETS = ['twitter', 'bbc']
SYNTHETIC = ['synthetic-random', 'synthetic-gaussian']
GRAPHS = []

def generate_name(dataset_name, trial, tree_type="ultrametric"):
    plt_name = '/home/sam/tree_learning/images/{dsname}/{tree}-{trial}-losses'.format(dsname=dataset_name, tree=tree_type, trial=trial)

    sf_name = '/data/sam/{dsname}/{tree}-{trial}'.format(dsname=dataset_name, tree=tree_type, trial=trial)
    return plt_name, sf_name

# parameters: distances, distributions, number to sample, number of processes
# returns: d1 index, d2 index, ot distance(d1, d2)
def generate_dataset(D, distributions, sample=None, n_processes = 20):
    num_distributions = len(distributions)
    if sample:
        pairs = np.random.randint(0, high=num_distributions, size=(sample, 2))
    else:
        pairs = []
        for i in range(num_distributions):
            for j in range(i + 1, num_distributions):
                pairs.append([i, j])
        pairs = np.array(pairs)
    
    pool = mp.Pool(processes = n_processes)
    jobs = []

    for pair in tqdm(pairs):
        d1 = distributions[pair[0]]
        d2 = distributions[pair[1]]
        combined_vec = d1 + d2
        non_zero_indices = np.nonzero(combined_vec)[0]
        small_D = D[non_zero_indices][:, non_zero_indices]
        job = pool.apply_async(ot.emd2, args=(d1[non_zero_indices], d2[non_zero_indices], small_D))
        jobs.append(job)
    
    pool.close()
    for job in tqdm(jobs):
        job.wait()
    d = [job.get() for job in jobs]
    return pairs[:, 0], pairs[:, 1], d

def train(dataset_name, distances, train_distributions,
          bsz = 1, TR = [], trial='DEBUG', sample=None, 
          device='cuda:0', tree_type='ultrametric', k=5, d = 6):
    # Construct tree
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    tree = None
    print("Constructing tree")
    if tree_type == 'ultrametric':
        tree = UltrametricOptimizedTree(distances, device=device)
    else:
        raise NameError("tree not implemented")
        return 0
    print("Finished constructing tree, transfering to device:", device)
    tree.to(device)
    
    # Initialize top level path
    top_level_pth = os.path.join('/data/sam', dataset_name, trial)
    log_path = os.path.join(top_level_pth, 'LOG')
    data_path = os.path.join(top_level_pth, 'data')
    if not os.path.exists(top_level_pth):
        os.mkdir(top_level_pth) # top level folder for experiment
        os.mkdir(log_path) # log path     
        os.mkdir(data_path) # data path

    # Initialize optimizer
    optimizer = torch.optim.Adam(tree.parameters(), lr = 0.01 )
    
    # check if dataset has already been generated
    dataset_fname = os.path.join(data_path, 'train_data.npz')
    if os.path.exists(dataset_fname):
        print("dataset generated, loading from saved files")
        train_dataset = np.load(dataset_fname)
        x = train_dataset['x']
        y = train_dataset['y']
        d = train_dataset['d']
    else:
        print("dataset not generated yet, generating dataset......")
        x, y, d = generate_dataset(distances, train_distributions, sample=sample)
        # save generated dataset
        np.savez(dataset_fname, x=x, y=y, d=d)

    # Configure dataset and dataloader
    dataset = DistanceDataset(torch.tensor(x, dtype=torch.long), 
                              torch.tensor(y, dtype=torch.long),  
                              torch.tensor(d))
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Configure file name of loss curves plot
    plt_fname = os.path.join(dataset_name, trial, 'training_curve')

    # Configure file name of final saved tree model
    final_tree_fname = os.path.join(top_level_pth, 'final_tree.pt')

    # Trees saved in epochs
    logged_trees = os.path.join(log_path, 'LOG')

    # train tree
    tree.train(dataloader,
                optimizer, 
                train_distributions, 
                max_iterations=50, 
                plt_name=plt_fname, 
                save_epoch=True, 
                bsz=bsz, 
                filename=logged_trees)
    
    # save final tree information
    torch.save({'parents': tree.parents,
                'leaves': tree.leaves,
                'subtree':tree.subtree,
                'M_idx': tree.M_idx,
                'weights': tree.np_parameters,
                'optimized_state_dict': tree.state_dict()}, 
                final_tree_fname)
    print("Saving tree")
    return tree

def main():
    parser = argparse.ArgumentParser(description="Process training parameters")
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--trial", type=str)
    parser.add_argument("--device", type=str)
    parser.add_argument("--numdist", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-points",type=int )
    parser.add_argument("--sampling", type=int)
    parser.add_argument("--sparsity", type=float, default=1.0)
    parser.add_argument("--sparse-word-dist", action='store_true', help='Add this flag with word dataset to use previously defined test/train splits')

    args = parser.parse_args()

    if args.dataset_name in WORD_DATASETS:
        filename = '/data/sam/{}/swmd.npz'.format(args.dataset_name)
        data = np.load(filename, allow_pickle=True)
        vecs = data['vectors']
        num_vecs = len(vecs)
        if args.sparse_dist:
            distributions = data['distributions']
            matfile='/data/sam/{name}/{name}-emd_tr_te_split.mat'.format(name=args.dataset_name)
            f = scipy.io.loadmat(matfile)
            TR = np.array(f['TR'])[0] - 1
            train_distributions = distributions[TR]
        else:
            assert args.numdist != None
            train_distributions = generate_uniform_dists(args.numdist, 
                                                         num_vecs, 
                                                         sparsity=args.sparsity)
    elif args.dataset_name in SYNTHETIC:
        assert args.num_points != None and args.numdist != None
        vecs = generate_random_points(args.num_points, dim=2)
        num_vecs = args.num_points
        train_distributions = generate_uniform_dists(args.numdist, num_vecs, sparsity=args.sparsity)
    else:
        print("Dataset not implemented!")
        return 0
    
    distances = ot.dist(vecs, vecs)


    # get overall distance matrix
    distances = ot.dist(vecs, vecs)

    tree = train(args.dataset_name, 
          distances, 
          train_distributions, 
          bsz=args.batch_size,
          trial=args.trial, 
          sample=args.sampling, 
          device=args.device, 
          tree_type=args.tree)
    
    if args.dataset_name in SYNTHETIC:
        vec_path = os.path.join('/data/sam', args.dataset_name, args.trial, 'points.npy')
        np.save(vec_path, vecs)

    return 0

def graph_dataset():
    parser = argparse.ArgumentParser(description="Process training parameters")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--vectors", type=str)
    parser.add_argument("--distances", type=str)
    parser.add_argument("--trial", type=str)
    parser.add_argument("--device", type=str)
    parser.add_argument("--numdist", type=int)
    args = parser.parse_args()

    
    vectors = []
    #vectors = generate_random_points(num_point, dim=2)
    distance_mat = np.load(args.distances)
    D = 1 - distance_mat
    n = distance_mat.shape[0]
    diag = np.arange(n)
    D[diag, diag] = 0
    distributions = generate_random_dists(args.numdist, distance_mat.shape[0])

    TR = np.arange(0, args.numdist)
        
    train(args.dataset, vectors, distributions, distance_mat=D, TR=TR, trial=args.trial, sample = False, device=args.device, bsz=3)

if __name__=="__main__":
    main()
