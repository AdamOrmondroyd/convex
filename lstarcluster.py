import numpy as np
from sklearn.cluster import KMeans


class LStarCluster:

    def __init__(self, likelihood, prior):
        self.likelihood = lambda x: likelihood(prior(x))[0]

    def __call__(self, position_matrix):
        print("L* clustering", flush=True)
        kmeans = KMeans(n_clusters=2, init='k-means++', n_init='auto')
        labels = kmeans.fit_predict(position_matrix)
        midpoint = (kmeans.cluster_centers_[0] +
                    kmeans.cluster_centers_[1]) / 2
        logL_0 = self.likelihood(kmeans.cluster_centers_[0])
        logL_1 = self.likelihood(kmeans.cluster_centers_[1])
        logL_mid = self.likelihood(midpoint)
        logL_star = np.min([self.likelihood(x) for x in position_matrix])
        # midpoint lies within L* so don't cluster
        if not ((logL_mid < logL_0) and (logL_mid < logL_1) and
                (logL_mid > logL_star)):
            print("Single mode!", flush=True)
            return np.zeros_like(labels)

        if labels[0] != 0:
            labels = 1 - labels
        return labels
