cimport cython
# from libc.stdlib cimport malloc, free

import numpy as np
from scipy.cluster.hierarchy import linkage
cimport numpy as np

cdef extern from "numpy/npy_math.h":
    cdef enum:
        NPY_INFINITYF

# assume leaves = [1, 2, ..... n]
def scipy_mst(double[:] vec, int n, np.ndarray[int, ndim=2] M_idx, np.ndarray[double, ndim=1] parameters, 
               np.ndarray[int, ndim=1] leaves, np.ndarray[int, ndim=1] parents):
    cdef double[:, :] Z = linkage(vec)
    cdef np.ndarray[np.uint8_t, ndim=2, cast=True] subtrees = np.zeros((2*n - 1, n), dtype=bool)
    cdef np.uint8_t true_val = 1
    subtrees[leaves, leaves - 1] = true_val
    parents[1: n + 1] = leaves
    M_idx[leaves - 1, leaves - 1] = leaves - 1
    cdef int lc_idx, rc_idx
    cdef int i
    cdef int parent_index
    for i in range(n - 1):
        lc_idx = int(Z[i][0]) + 1
        rc_idx = int(Z[i][1]) + 1
        parent_index = int(n + i + 1)
        if parent_index == 2 * n - 1:
            parent_index = int(0)
        parents[lc_idx] = parent_index
        parents[rc_idx] = parent_index
        subtrees[parent_index] = subtrees[lc_idx] + subtrees[rc_idx]
        #subtrees[parent_index, subtrees[lc_idx]] = true_val
        #subtrees[parent_index, subtrees[rc_idx]] = true_val
        M_idx[np.ix_(subtrees[lc_idx], subtrees[rc_idx])] = int(n + i)
        M_idx[np.ix_(subtrees[rc_idx], subtrees[lc_idx])] = int(n + i)
        parameters[n + i] = Z[i][2]
    
    parents[0] = -1
    return subtrees

def gpu_mst(double[:] vec, int n, np.ndarray[int, ndim=2] M_idx, np.ndarray[double, ndim=1] parameters, 
               np.ndarray[int, ndim=1] leaves, np.ndarray[int, ndim=1] parents):
    cdef double[:, :] Z = linkage(vec)
    cdef np.ndarray[np.uint8_t, ndim=2, cast=True] subtrees = np.zeros((2*n - 1, n), dtype=bool)
    cdef np.uint8_t true_val = 1
    subtrees[leaves, leaves] = true_val
    parents[: n ] = leaves
    M_idx[leaves, leaves] = leaves

    cdef int lc_idx, rc_idx
    cdef int i
    cdef int parent_index

    for i in range(n - 1):
        lc_idx = int(Z[i][0])
        rc_idx = int(Z[i][1])
        parent_index = n + i
        parents[lc_idx] = parent_index
        parents[rc_idx] = parent_index
        subtrees[parent_index] = subtrees[lc_idx] + subtrees[rc_idx]
        M_idx[np.ix_(subtrees[lc_idx], subtrees[rc_idx])] = int(n + i)
        M_idx[np.ix_(subtrees[rc_idx], subtrees[lc_idx])] = int(n + i)
        parameters[n + i] = Z[i][2]

    parents[2*n - 2] = -1
    return subtrees