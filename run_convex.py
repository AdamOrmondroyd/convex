"""
MWE showing the different behaviour of polychord vs multinest in double Gaussian case.
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
from scipy.special import logsumexp
import anesthetic as ac
import mpi4py

### polychord ###
import pypolychord
from pypolychord.priors import UniformPrior
from sklearn.metrics import silhouette_score

from convexcluster import ConvexCluster
from lstarcluster import LStarCluster
from convexspectralcluster import ConvexAdjacencyCluster


nDims = 2
nDerived = 0
sigma = 0.05
nlive = 200

centres = np.array([[-0.5] * 2 + [0] * (nDims-2), [0.5] * 2 + [0] * (nDims-2)])
weights = np.array([1/2, 1/2])
# centres = np.array([[-0.6] * nDims, [-0.4] * nDims, [0.4] * nDims, [0.6] * nDims])
# centres = np.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]])
# weights = np.array([0.25, 0.25, 0.25, 0.25])
# centres = np.array([[-0.5] * nDims, [0.0] * nDims, [0.5] * nDims])
# centres = np.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]])
# weights = np.array([1/3, 1/3, 1/3])


def gaussian(theta):
    """Double Gaussian likelihood."""
    nDims = len(theta)
    logL = -np.log(2 * np.pi * sigma * sigma) * nDims / 2
    logL += logsumexp(
        -np.sum((theta - centres) ** 2, axis=-1) / 2 / sigma / sigma,
        b=weights)
    return logL, []


def rosenbrock(theta):
    """Fix the rosenbrock likelihood"""
    a = 1
    b = 100
    logL = - sum(b * (theta[1:] - theta[:-1]**2)**2 + (a - theta[:-1])**2)
    return logL, []


likelihood = rosenbrock


def polychord_prior(hypercube):
    """Uniform prior from [-1,1]^D."""
    return UniformPrior(-1, 1)(hypercube)


convex_cluster = ConvexCluster(likelihood, polychord_prior)
lstar_cluster = LStarCluster(likelihood, polychord_prior)
convex_adjacency_cluster = ConvexAdjacencyCluster(likelihood, polychord_prior)

fig = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(3, 3, fig, wspace=0, hspace=0)
for i in range(9):
    print(f"iteration {i}", flush=True)
    cornerfig, cornerax = ac.make_2d_axes([0, 1],
                                          subplot_spec=gs[i // 3, i % 3],
                                          fig=fig)
    output = pypolychord.run(likelihood, nDims, prior=polychord_prior,
                             file_root="native", read_resume=False,
                             nlive=nlive,
                             )
    output.plot_2d(cornerax, label="KNN")
    output = pypolychord.run(likelihood, nDims, prior=polychord_prior,
                             file_root="convex_adjacency_cluster", read_resume=False,
                             cluster=convex_adjacency_cluster,
                             nlive=nlive,
                             )
    output.plot_2d(cornerax, label="convex")

    if i == 2:
        cornerax.iloc[0, 1].legend()
    if i % 3 != 0:
        for ax in cornerax:
            for a in cornerax[ax]:
                a.set(yticklabels=[], ylabel="")
    if i / 3 < 2:
        for ax in cornerax:
            for a in cornerax[ax]:
                a.set(xticklabels=[], xlabel="")
plt.show()
