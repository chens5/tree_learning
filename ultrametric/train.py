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

def document_classification(dataset_name, word_vecs, documents, distance_mat = [], TR=[], trial=0, sample=False, device="cuda:1", bsz=-1):
    print("Generating distance matrix.....")
    if len(distance_mat) == 0:
        D = ot.dist(word_vecs, word_vecs, metric='euclidean')
        word_vecs = word_vecs
    else:
        D = distance_mat
    
    # split into test/train
    if len(TR) != 0:
        X_train = TR
    else:
        X_train, X_test, y_train, y_test = train_test_split(np.arange(0, len(documents)), np.zeros(len(documents)), test_size=0.30, random_state=5)
    x = []
    y = []
    d = []

    train_distributions = documents[X_train]

    #train_distributions = documents
    
    print("Number of training distributions", train_distributions.shape[0])
    print("Generating dataset......")
    pool = mp.Pool(processes=20)
    jobs = []
    if sample==True:
        
        xvals = np.random.choice(len(train_distributions), 1225)
        yvals = np.random.choice(len(train_distributions), 1225)
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

    print("number of pairs", len(x))
    dataset = DistanceDataset(torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long),  torch.tensor(d))
    dataloader = DataLoader(dataset, batch_size=225, shuffle=True)
    print("Beginning to construct tree..........")
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    tree = UltrametricOptimizedTree(D, device=device)
    tree.to(device)

    optimizer = torch.optim.Adam(tree.parameters(), lr = 0.1 )

    print("Starting training............")
    plt_name =dataset_name + '_losses_50epochs_t' + str(trial)
    save_file = "/data/sam/" + dataset_name + "/results/" + str(trial) 
    start = time.time()
    tree.train(dataloader, optimizer, train_distributions, max_iterations=50, plt_name=plt_name, save_epoch=True, bsz=bsz, filename=save_file)
    end = time.time()
    print("TIME FOR TRAINING:", end - start)
    save_file = "/data/sam/" + dataset_name + "/results/" + str(trial) + ".pt"
    print("Saved to:", save_file)
    torch.save({'parents': tree.parents,
        'leaves': tree.leaves,
                'subtree':tree.subtree,
                'vec': tree.vec,
                'M_idx': tree.M_idx,
                'np_parameters': tree.np_parameters,
                'optimized_state_dict': tree.state_dict()}, 
                save_file)

    return tree

def synthetic_dataset():
    parser = argparse.ArgumentParser(description="Process training parameters")
    parser.add_argument("--numpoints", type=int)
    parser.add_argument("--dimensions", type=int)
    parser.add_argument("--savedataset", type=bool)
    parser.add_argument("--numdist", type=int)
    parser.add_argument("--trial", type=int)
    parser.add_argument("--device", type=str)
    args = parser.parse_args()

    num_points = [800, 1000, 1500]
    for num_point in num_points:

        vectors = generate_random_points(num_point, dim=2)
        distributions = generate_random_dists(args.numdist, num_point)
        print(vectors.shape)
        print(distributions.shape)
        
        if args.savedataset:
            sf_vec_name = '/data/sam/synthetic/vectors_num' + str(num_point) + '.npy'
            np.save(sf_vec_name, vectors)
        #training_size = int(args.numdist*0.80)
        #TR = np.random.randint(0, high=args.numdist, size=training_size)
        TR = np.arange(0, args.numdist)
        
        document_classification('synthetic', vectors, distributions, TR=TR, trial=num_point, sample = False, device=args.device, bsz=1)
    
def graph_dataset():
    parser = argparse.ArgumentParser(description="Process training parameters")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--vectors", type=str)
    parser.add_argument("--distances", type=str)
    parser.add_argument("--trial", type=int)
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
        
    document_classification(args.dataset, vectors, distributions, distance_mat=D, TR=TR, trial=args.trial, sample = False, device=args.device, bsz=3)

def main():
    #image_classification()
    parser = argparse.ArgumentParser(description="Process training parameters")
    parser.add_argument("--dataset_name", type=str, help="Dataset for training")
    parser.add_argument("--trial", type=int, help="trial id")
    parser.add_argument("--is_sampling", type=bool, help="Sample pairs from training dataset.", default=False)
    parser.add_argument("--device", type=str, help="GPU")
    parser.add_argument("--labels", type=str, help="labels for distributions")
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--npz", type=str)
    parser.add_argument('--matfile', type=str)
    #parser.add_argument("--save_file", type=str, help="save file name")
    args = parser.parse_args()
    # f = open("/data/sam/twitter/results/"+str(args.trial) + "specifications.txt", 'w')
    # f.write(args)

    # Load dataset
    print("Loading dataset")
    save_file = np.load(args.npz, allow_pickle=True)
    word_vecs = save_file['vectors']
    documents = save_file['distributions']
    #documents = generate_random_dists(100, word_vecs.shape[0])
    f = scipy.io.loadmat(args.matfile)
    #TR = []
    #TR = np.arange(100)
    TR = np.array(f['TR'])[0] - 1

    
    print("Starting training for", args.dataset_name)
    print("Sampling pairs:", args.is_sampling)
    document_classification(args.dataset_name, word_vecs, documents, TR=TR, trial=args.trial, sample = args.is_sampling, device=args.device, bsz=250)

if __name__=="__main__":
    #main()
    synthetic_dataset()
    #graph_dataset()
