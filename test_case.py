"""
This script demonstrates how can we implement smooth k-mean on toy dataset
"""

import numpy as np
from main import smooth_kmeans

# Generate some toy data
np.random.seed(0)  # for reproducibility
X = np.random.randn(100, 2)  # 100 samples with 2 features

# Set up options
options = {
    'Distance': 'sqeuclidean',  # you can choose 'sqeuclidean' or 'cosine'
    'SmoothMethod': 'Boltzmann',  # you can choose 'wcss', 'logsumexp', 'p-norm', or 'Boltzmann'
    'SmoothCoefficient': 'dvariance',  # you can specify a float or use 'dvariance'
    'MaxIter': 500,  # maximum number of iterations
    'Eta': 1e-3,  # tolerance for convergence
    'Replicates': 1,  # number of replicates
    'Start': 'plus'  # method for specifying centroid initialization
}

# Number of clusters
K = 3

# Fit the smooth k-means model to the toy data
idx, C, numit, smoothness, W, sumd, D, J = smooth_kmeans(X, K, options)

# Print results
print("Cluster indices:", idx)
print("Centroids:", C)
print("Number of iterations:", numit)
print("Smoothness parameter:", smoothness)
print("Weights:", W)
print("Within-cluster sums of point-to-centroid distances:", sumd)
print("Distances from each point to every centroid:", D)
print("Lowest loss value:", J)
