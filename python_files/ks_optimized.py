################################################################################
# Kennard-Stone Efficient Sampling Implementation
# Source: https://github.com/ajz34/Kennard-Stone-Efficient
# Author: Zhenyu Zhu (ajz34)
# Licensed under MIT License - See THIRD_PARTY_LICENSES.txt

#
# Modified: Added explanatory comments for clarity (original code unchanged)
################################################################################
import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool

def ks_sampling_mem(X, seed=None, n_result=None, n_proc=4, n_batch=1000):
    """
    Memory-efficient Kennard-Stone sampling for large datasets.
    
    Computes distances in batches to avoid memory overflow when dealing with
    very large sampling spaces. Uses parallel processing to mitigate the 
    computational overhead of slice-wise distance calculations.
    
    Args:
        X (array): Sample matrix (n_samples, n_features)
        seed (array, optional): Initial seed points. If None, finds most distant pair
        n_result (int, optional): Number of samples to select. Default is all samples
        n_proc (int): Number of parallel processes. Default is 4
        n_batch (int): Batch size for distance computation. Default is 1000
    
    Returns:
        tuple: (result_indices, distances) - Selected sample indices and their distances
    
    Note:
        - Uses slice-wise distance computation to reduce memory footprint
        - Parallelizes batch computations across n_proc processors
        - Slower than classical KS but handles datasets that don't fit in memory
    """
    import numpy as np  # Ensure numpy is imported inside the function
    X = np.asarray(X, dtype=float)
    n_sample = X.shape[0]
    if n_result is None:
        n_result = X.shape[0]
    
    if seed is None or len(seed) == 0:
        # Calculate squared norms for efficient distance computation
        t = np.einsum("ia, ia -> i", X, X)
        
        def get_dist_slice(sliceA, sliceB):
            """Compute distance matrix between two slices of samples."""
            import numpy as np  # Ensure numpy is imported inside the nested function
            # Distance formula: ||a-b||^2 = ||a||^2 - 2*a·b + ||b||^2
            distAB = t[sliceA, None] - 2 * X[sliceA] @ X[sliceB].T + t[None, sliceB]
            if sliceA == sliceB:
                np.fill_diagonal(distAB, 0)  # Distance to self is zero
            return np.sqrt(distAB)
        
        def get_maxloc_slice(slice_pair):
            """Find maximum distance and its location within a slice pair."""
            dist_slice = get_dist_slice(slice_pair[0], slice_pair[1])
            max_indexes = np.unravel_index(np.argmax(dist_slice), dist_slice.shape)
            return dist_slice[max_indexes], max_indexes[0] + slice_pair[0].start, max_indexes[1] + slice_pair[1].start
        
        # Divide sample space into batches
        p = list(np.arange(0, n_sample, n_batch)) + [n_sample]
        slices = [slice(p[i], p[i+1]) for i in range(len(p) - 1)]
        # Generate all unique slice pairs (including self-pairs)
        slice_pairs = [(slices[i], slices[j]) for i in range(len(slices)) for j in range(len(slices)) if i <= j]
        
        # Parallel computation of max distances in each slice pair
        with Pool(n_proc) as p:
            maxloc_slice_list = p.map(get_maxloc_slice, slice_pairs)
        # Find overall maximum distance across all slices
        max_indexes = maxloc_slice_list[np.argmax([v[0] for v in maxloc_slice_list])][1:]
        seed = max_indexes
    seed = np.asarray(seed, dtype=np.uintp)
    
    return ks_sampling_mem_core(X, seed, n_result)

###################################################################################################################################################
def ks_sampling_mem_core(X, seed, n_result):
    """
    Core Kennard-Stone algorithm with memory-efficient implementation.
    
    Iteratively selects samples that are maximally distant from already selected ones.
    Maintains only the minimum distances to reduce memory usage.
    
    Args:
        X (array): Sample matrix
        seed (array): Initial seed point indices
        n_result (int): Total number of samples to select
    
    Returns:
        tuple: (result_indices, distances) - Selected indices and their minimum distances
    """
    result = np.zeros(n_result, dtype=int)
    v_dist = np.zeros(n_result, dtype=float)
    n_seed = len(seed)
    n_sample = X.shape[0]

    def sliced_dist(idx):
        """Calculate distances from a given point to all remaining points."""
        tmp_X = X[remains] - X[idx]
        return np.sqrt(np.einsum("ia, ia -> i", tmp_X, tmp_X))

    # Initialize with points not in seed
    remains = [i for i in range(n_sample) if i not in seed]
    result[:n_seed] = seed
    
    # Calculate initial distance between seed points if there are exactly 2
    if n_seed == 2:
        v_dist[0] = np.linalg.norm(X[seed[0]] - X[seed[1]])
    
    # Calculate minimum distances to seed points
    min_vals = sliced_dist(seed[0])
    for n in seed:
        np.min(np.array([min_vals, sliced_dist(n)]), axis=0, out=min_vals)
    
    # Iteratively select points with maximum minimum distance
    for n in range(n_seed, n_result):
        # Find point with largest minimum distance to selected set
        sup_index = min_vals.argmax()
        result[n] = remains[sup_index]
        v_dist[n - 1] = min_vals[sup_index]
        
        # Remove selected point from remaining candidates
        remains.pop(sup_index)
        min_vals[sup_index:-1] = min_vals[sup_index + 1:]
        min_vals = min_vals[:-1]
        
        # Update minimum distances with newly selected point
        np.min(np.array([min_vals, sliced_dist(result[n])]), axis=0, out=min_vals)
    
    return result, v_dist