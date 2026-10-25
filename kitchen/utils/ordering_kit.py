from scipy.cluster.hierarchy import linkage, leaves_list, optimal_leaf_ordering
from scipy.spatial.distance import pdist
import numpy as np


def linkage_order(X: np.ndarray, method="ward", metric="euclidean"):
    
    X[np.isinf(X)] = np.nan
    X[np.isnan(X)] = 0    

    d = pdist(X, metric=metric)

    if np.isnan(d).any() or np.isinf(d).any():
        print(f"Linkage NaN distances: {np.isnan(d).sum()} out of {len(d)}, Inf distances: {np.isinf(d).sum()} out of {len(d)}")
    
    Z = linkage(d, method=method)
    Z = optimal_leaf_ordering(Z, d)
    return leaves_list(Z)


def linkage_leaf_maps(linkage, n_leaf: int) -> dict:
    leaf_map = {i: {i} for i in range(n_leaf)}
    for i, row in enumerate(linkage):
        node_id = n_leaf + i
        child1, child2 = int(row[0]), int(row[1])
        leaf_map[node_id] = leaf_map[child1].union(leaf_map[child2])
    return leaf_map


