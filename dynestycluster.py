"""
Clustering algorithm copied from `dynesty`

Note that scipy seems to number clusters from 1.
"""

import numpy as np
from scipy import spatial, cluster

# note: scipy seems to number clusters from 1


def relabel(labels):
    """
    Relabel cluster labels so that they are in ascending order.
    """

    appearance_order = {}
    num_found = 0

    for label in labels:
        if label not in appearance_order:
            appearance_order[label] = num_found
            num_found += 1

    for i, old_label in enumerate(labels):
        labels[i] = appearance_order[old_label]
    return labels


def dynesty_cluster(points, depth=0):
    """Compute covariance from re-centered clusters."""

    print(f"Dynesty clustering {depth=}", flush=True)
    # Compute pairwise distances.
    try:
        inv = np.linalg.inv(np.cov(points.T)).T
    except np.linalg.LinAlgError as e:
        print(e.message, flush=True)
        print("assuming single cluster and moving on", flush=True)
        return np.zeros(len(points))
    distances = spatial.distance.pdist(points,
                                       metric='mahalanobis',
                                       # VI=self.am)  # see what happens with default for now
                                       VI=inv,
                                       )

    # Identify conglomerates of points by constructing a linkage matrix.
    linkages = cluster.hierarchy.single(distances)

    # Cut when linkage between clusters exceed the radius.
    labels = cluster.hierarchy.fcluster(linkages,
                                        1.0,
                                        criterion='distance')
    labels = relabel(np.array(labels) - 1)
    print(f"{labels=}", flush=True)
    if max(labels) > 0:
        print("it's morbin' time")
        labels = combine_labels(labels, *[dynesty_cluster(
            points[labels == label], depth=depth+1)
            for label in np.unique(labels)])
        print("escaped", flush=True)
    return labels


def combine_labels(initial_splitting, *labelss):
    """
    Combine labells from recursive clustering into a single list
    """
    print("Combining labels", flush=True)
    print(f"{labelss=}", flush=True)
    print(f"{initial_splitting=}", flush=True)
    sizes = [max(labels) for labels in labelss]

    for i in np.arange(max(initial_splitting-1), -1, -1):
        print(f"{sum(sizes[:i-1])=}", flush=True)
        initial_splitting[initial_splitting == i] = labelss[i] + sum(sizes[:i-1])
    print(f"{initial_splitting=}", flush=True)
    return initial_splitting
