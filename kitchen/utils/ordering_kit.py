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