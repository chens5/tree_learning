cimport cython
# from libc.stdlib cimport malloc, free

import numpy as np
from scipy.cluster.hierarchy import linkage
cimport numpy as np

cdef extern from "numpy/npy_math.h":
    cdef enum:
        NPY_INFINITYF


def scipy_mst(double[:] vec, int n, np.ndarray[int, ndim=2] M_idx, np.ndarray[double, ndim=1] parameters, 
               np.ndarray[int, ndim=1] leaves, np.ndarray[int, ndim=1] parents):
    cdef double[:, :] Z = linkage(vec)
    cdef np.ndarray[np.uint8_t, ndim=2, cast=True] subtrees = np.zeros((2*n - 1, n), dtype=bool)
    cdef np.uint8_t true_val = 1
    subtrees[leaves, leaves] = true_val
    parents[:n] = leaves
    cdef int lc_idx, rc_idx
    for i in range(n - 1):
        lc_idx = int(Z[i][0])
        rc_idx = int(Z[i][1])
        parents[lc_idx] = n + i
        parents[rc_idx] = n + i
        subtrees[n + i, subtrees[lc_idx]] = true_val
        subtrees[n + i, subtrees[rc_idx]] = true_val
        M_idx[np.ix_(subtrees[lc_idx], subtrees[rc_idx])] = int(n + i)
        M_idx[np.ix_(subtrees[rc_idx], subtrees[lc_idx])] = int(n + i)
        parameters[n + i] = Z[i][2]
    
    parents[2*n - 2] = -1
    return parents