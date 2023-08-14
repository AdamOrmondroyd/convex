import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class ConvexCluster:

    def __init__(self, likelihood, prior):
        self.likelihood = lambda x: likelihood(prior(x))[0]

    def __call__(self, position_matrix):
        print("Likelihood clustering", flush=True)
        kmeans = KMeans(n_clusters=2, init='k-means++', n_init='auto')
        labels = kmeans.fit_predict(position_matrix)
        if silhouette_score(position_matrix, labels) <= 0:
            print("Single mode from silhouette score!", flush=True)
            return np.zeros_like(labels)
        midpoint = (kmeans.cluster_centers_[0] +
                    kmeans.cluster_centers_[1]) / 2
        logL_0 = self.likelihood(kmeans.cluster_centers_[0])
        logL_1 = self.likelihood(kmeans.cluster_centers_[1])
        logL_mid = self.likelihood(midpoint)
        if logL_mid > logL_0 or logL_mid > logL_1:
            print("Single mode from convexity!", flush=True)
            return np.zeros_like(labels, dtype=int)
        subcluster_0 = self.__call__(position_matrix[labels == 0])
        subcluster_1 = self.__call__(position_matrix[labels == 1])
        labels[labels == 1] = subcluster_1 + np.max(subcluster_0) + 1
        labels[labels == 0] = subcluster_0
        print(f"{labels=}", flush=True)
        return labels


class DynamicConvexCluster:

    def __init__(self, likelihood, prior):
        self.likelihood = lambda x: likelihood(prior(x))[0]

    def __call__(self, position_matrix):
        print("Likelihood clustering", flush=True)
        for k in range(2, ):
            kmeans = KMeans(n_clusters=k, init='k-means++', n_init='auto')
            labels = kmeans.fit_predict(position_matrix)
            if silhouette_score(position_matrix, labels) <= 0:
                print("Single mode from silhouette score!", flush=True)
                return np.zeros_like(labels)
            midpoint = (kmeans.cluster_centers_[0] +
                        kmeans.cluster_centers_[1]) / 2
            logL_0 = self.likelihood(kmeans.cluster_centers_[0])
            logL_1 = self.likelihood(kmeans.cluster_centers_[1])
            logL_mid = self.likelihood(midpoint)
            if logL_mid > logL_0 or logL_mid > logL_1:
                print("Single mode from convexity!", flush=True)
                return np.zeros_like(labels, dtype=int)
            subcluster_0 = self.__call__(position_matrix[labels == 0])
            subcluster_1 = self.__call__(position_matrix[labels == 1])
            labels[labels == 1] = subcluster_1 + np.max(subcluster_0) + 1
            labels[labels == 0] = subcluster_0
            print(f"{labels=}", flush=True)
        return labels
