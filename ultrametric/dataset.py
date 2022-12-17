import scipy.io

import argparse
import numpy as np

from tqdm import tqdm, trange

parser = argparse.ArgumentParser(description="Process datasets")
parser.add_argument("--datafile", type=str, help=".mat data file")
parser.add_argument("--savefile", type=str)
args = parser.parse_args()


f = scipy.io.loadmat(args.datafile)
# distributions
X = np.array(f['X'])[0]
# labels
Y = np.array(f['Y'])[0]
# BOW_X, distributions
BOW_X = np.array(f['BOW_X'])[0]
# words
words = np.array(f['words'])[0]
clean_words = []
for i in range(X.shape[0]):
    document_words = []
    cur_words = words[i][0]
    for j in range(X[i].shape[1]):
        document_words.append([cur_words[j][0]])
    clean_words.append([document_words])
words = np.array(clean_words)

# TR, TE are 5 train-test splits
TR = np.array(f['TR'])
TE = np.array(f['TE'])

# Get words and word vectors
num_distributions = X.shape[0]
word_dictionary = {}
all_words = []
all_word_vectors = []
all_indices = []

index = 0
for ii in range(num_distributions):
    words_in_distribution = words[ii][0]
    vectors_in_distribution = X[ii]
    sz = len(words_in_distribution)
    idxs = []

    if sz != vectors_in_distribution.shape[1]:
        print(sz, vectors_in_distribution.shape[1])
        raise Exception("Error in dataset, there must be an equal number of words in distribution as vectors")
    
    for i in range(sz):
        word = words_in_distribution[i][0]
        if word not in word_dictionary:
            word_dictionary[word] = index
            all_words.append(word)
            all_word_vectors.append(vectors_in_distribution[:, i])
            index += 1 
        idxs.append(word_dictionary[word])
    all_indices.append(idxs)



print("Number of words:", index)

# Get all distributions
print("Getting all distributions......")
distributions = np.zeros((num_distributions, index))
for ii in trange(len(all_indices)):
    indices = all_indices[ii]
    bow_x = BOW_X[ii][0]
    a = bow_x/bow_x.sum()
    distributions[ii, indices] = a


print("Number of distributions:" ,num_distributions)
np.savez(args.savefile, distributions=distributions, vectors=all_word_vectors,words=all_words, indices = all_indices)