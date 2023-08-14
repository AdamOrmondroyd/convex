import numpy as np
import networkx as nx
from networkx.algorithms import community


class ConvexAdjacencyCluster:

    def __init__(self, likelihood, prior):
        # since position matrix is in unit space, have to inverse prior
        # also have to take first element of PolyChord likelihood
        self.likelihood = lambda theta: likelihood(prior(theta))[0]

    def adjacency_matrix(self, position_matrix):
        logL = np.array([self.likelihood(p) for p in position_matrix])
        midpoints = (position_matrix + position_matrix[:, np.newaxis]) / 2
        logL_midpoints = np.array(
            [[self.likelihood(ab) for ab in a] for a in midpoints])

        return np.logical_and(logL_midpoints > logL[:, np.newaxis],
                              logL_midpoints > logL, True, False)

    def __call__(self, position_matrix):
        print("Convex adjacency clustering", flush=True)
        g = nx.Graph(self.adjacency_matrix(position_matrix))
        print("a", flush=True)
        labels = community.label_propagation_communities(g)
        return labels
