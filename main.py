import numpy as np
"""
This script shows the implementation of Smooth-k-mean
"""
def smooth_kmeans(X, K, options):
    """
    Inputs:
    X: numpy array. N-by-P data matrix containing N samples with P features.
    K: int. Number of clusters.
    options:
        - Distance: 'sqeuclidean' (default) | 'cosine'. Method for defining similarity.
          'sqeuclidean': squared euclidean distance. 'cosine': cosine
        - SmoothMethod: 'wcss' (default) | 'logsumexp' | 'p-norm' | 'Boltzmann'. Method for specifying smoothing functions.
          'wcss': within-sum-of-squared (hard k-means). 'logsumexp': the logsumexp function (MEFC).
          'p-norm': the p-norm function (fuzzy  k-means). 'Boltzmann': the Boltzmann operator (EKM).
        - SmoothCoefficient: float (default 1) | "dvariance". Smoothing parameter or method to calculate the smoothing parameter.
          'dvariance': data variance.
        - MaxIter: 500 (default) | int. Maximum number of iterations.
        - Eta: 1e-3 | float. Tolerance for convergence
        - Replicates: 1 (default) | int. Replicate number. The replication with the lowest objetive value will be chosen as the final outcome.
        - Start: 'plus' (default) | numpy array. Method for specifying centroids initialization | K-by-P matrix containing initial centroids
          'plus': the k-means++ algorithm

    Outputs:
    idx: numpy array. The cluster indices in an N-by-1 vector
    C: numpy array. Estimated centroids in an K-by-P matrix
    W: numpy array. Weights in N-by-K matrix.
    sumd: numpy array. Within-cluster sums of point-to-centroid distances in the K-by-1 vector
    D: numpy array. Distances from each point to every centroid in the N-by-K matrix.
    J: float. The lowest loss value over replications.
    """

    # Arguments validation
    options.setdefault('Distance', 'sqeuclidean')
    options.setdefault('SmoothMethod', 'wcss')
    options.setdefault('SmoothCoefficient', 1)
    options.setdefault('MaxIter', 500)
    options.setdefault('Eta', 1e-3)
    options.setdefault('Replicates', 1)
    options.setdefault('Start', 'plus')

    if isinstance(options['Start'], np.ndarray):
        R = options['Start'].shape[2]
    else:
        R = options['Replicates']

    # Define distance function
    if options['Distance'] == 'sqeuclidean':
        dist = sqeuclidean
    elif options['Distance'] == 'cosine':
        dist = cosine

    # Define weight function
    if options['SmoothMethod'] == 'wcss':
        membership = wcss_membership
    elif options['SmoothMethod'] == 'logsumexp':
        membership = lse_membership
    elif options['SmoothMethod'] == 'p-norm':
        membership = pn_membership
    elif options['SmoothMethod'] == 'Boltzmann':
        membership = boltzmann_membership

    if isinstance(options['SmoothCoefficient'], str):
        if options['SmoothCoefficient'] == 'dvariance':
            options['SmoothCoefficient'] = 2 / np.mean(dist(X, np.mean(X, axis=0)))

    SmoothCoefficient = options['SmoothCoefficient']
    J_set = np.zeros(R)
    C_set = np.random.randn(K, X.shape[1], R)
    numit_set = np.zeros(R, dtype=int)
    SmoothCoefficient_set = np.zeros(R)

    # Start replicates
    for r in range(R):
        # Initialize centroid
        if isinstance(options['Start'], np.ndarray):
            C = options['Start'][:, :, r]  # K-by-P matrix
        else:
            if options['Start'] == 'plus':  # K-means++
                tmp = X.T  # P-by-Q matrix
                C = tmp[:, np.random.randint(tmp.shape[1])][:, np.newaxis]
                L = np.ones(tmp.shape[1], dtype=bool)
                for i in range(1, K):
                    D = tmp - C[:, L]
                    D = np.cumsum(np.sqrt(np.sum(D ** 2, axis=0)))
                    if D[-1] == 0:
                        C[:, i:] = tmp[:, np.tile(np.arange(i, K), (tmp.shape[1], 1)).T]
                        break
                    C[:, i] = tmp[:, np.argmax(np.random.rand() < D / D[-1])]
                    L = np.any(np.abs(C.T - tmp) > 1e-10, axis=1)
                C = C.T  # K-by-P matrix

        # Find Centroid
        C_old = C.copy()
        it = 1
        W = np.ones((X.shape[0], K))
        while True:
            D = dist(X, C)
            W = membership(D, SmoothCoefficient)
            for k in range(K):
                sum_Wk = np.sum(W[:, k])
                if sum_Wk == 0:  # which means there is one centroid too far from data so all membership to that centroid is zero
                    sum_Wk = np.finfo(float).eps  # to prevent NaN
                C[k, :] = np.sum(W[:, k, np.newaxis] * X, axis=0) / sum_Wk  # update k-th centroid

            if np.linalg.norm(C - C_old, 'fro') / np.linalg.norm(C, 'fro') < options['Eta']:
                break

            if it + 1 > options['MaxIter']:
                print(f"Failed to converge in {options['MaxIter']} iterations during replicate {r} for {options['SmoothMethod']} with {K} clusters")
                break
            else:
                C_old = C.copy()
                it += 1

        # Calculate objectives
        D = dist(X, C)
        if options['SmoothMethod'] == 'wcss':
            J = np.sum(np.min(D, axis=1))
        elif options['SmoothMethod'] == 'logsumexp':
            lambda_ = SmoothCoefficient
            J = -1 / lambda_ * np.sum(np.log(np.sum(np.exp(-lambda_ * D), axis=1)))
        elif options['SmoothMethod'] == 'p-norm':
            p = SmoothCoefficient
            J = np.sum(np.sum(D ** (-p), axis=1) ** (-1 / p))
        elif options['SmoothMethod'] == 'Boltzmann':
            alpha = SmoothCoefficient
            J = np.sum(np.sum(D * np.exp(-alpha * D), axis=1) / np.sum(np.exp(-alpha * D), axis=1))

        J_set[r] = J
        C_set[:, :, r] = C
        numit_set[r] = it
        SmoothCoefficient_set[r] = SmoothCoefficient

    best_id = np.argmin(J_set)
    J = J_set[best_id]
    C = C_set[:, :, best_id]
    numit = numit_set[best_id]
    smoothness = SmoothCoefficient_set[best_id]
    D = dist(X, C)
    idx = np.argmin(D, axis=1)
    sumd = np.zeros(K)
    for k in range(K):
        sumd[k] = np.sum(D[idx == k, k])
    W = membership(D, SmoothCoefficient)

    return idx, C, numit, smoothness, W, sumd, D, J

def sqeuclidean(X, C):
    """
    Input:
        X: N-by-P numpy array containing N instances.
        C: K-by-P numpy array containing K centroids.
    Output:
        D: N-by-K numpy array containing distances between data points and centroids
    """
    Q, _ = X.shape
    K, _ = C.shape
    D = np.zeros((Q, K))
    for k in range(K):
        c = C[k, :]
        D[:, k] = 0.5 * np.sum((X - c) ** 2, axis=1)
    return D

def cosine(X, C):
    """
    Input:
        X: N-by-P numpy array containing N instances.
        C: K-by-P numpy array containing K centroids.
    Output:
        D: N-by-K numpy array containing distances between data points and centroids
    """
    Q, _ = X.shape
    K, _ = C.shape
    D = np.zeros((Q, K))
    for k in range(K):
        c = C[k, :]
        D[:, k] = 1 - np.sum(X * c, axis=1) / (np.linalg.norm(X, axis=1) * np.linalg.norm(c))
    return D

def wcss_membership(D, _):
    """
    Input:
        D: N-by-K numpy array containing defined distance matrix (From N data points to K centeroids).
        _: Unused parameter.
    Output:
        W: N-by-K numpy array containing weights
    """
    K = D.shape[1]
    pos = np.argmin(D, axis=1)
    W = np.zeros_like(D, dtype=bool)
    W[np.arange(len(W)), pos] = True
    return W

def lse_membership(D, lambda_):
    """
    Input:
        D: N-by-K numpy array containing defined distance matrix (From N data points to K centeroids).
        lambda_: Smoothing parameter
    Output:
        W: N-by-K numpy array containing weights
    """
    K = D.shape[1]
    W = np.zeros_like(D)
    for k in range(K):
        W[:, k] = np.exp(-lambda_ * D[:, k]) / np.sum(np.exp(-lambda_ * D), axis=1)

    # Prevent all 0 membership because of numerical resolution
    zero_idx = np.where(np.sum(W, axis=1) == 0)[0]
    pos = np.argmin(D[zero_idx, :], axis=1)
    W[zero_idx, :] = 0
    W[zero_idx, pos] = 1

    return W

def pn_membership(D, p):
    """
    Input:
        D: N-by-K numpy array containing defined distance matrix (From N data points to K centeroids).
        p: Smoothing parameter
    Output:
        W: N-by-K numpy array containing weights
    """
    K = D.shape[1]
    W = np.zeros_like(D)
    D = D + np.finfo(float).eps  # Preventing NAN
    for k in range(K):
        W[:, k] = D[:, k] ** (-p - 1) / np.sum(D ** (-p), axis=1) ** (1 / (p + 1))

    # Prevent all 0 membership because of numerical resolution
    zero_idx = np.where(np.sum(W, axis=1) == 0)[0]
    pos = np.argmin(D[zero_idx, :], axis=1)
    W[zero_idx, :] = 0
    W[zero_idx, pos] = 1

    return W

def boltzmann_membership(D, alpha):
    """
    Input:
        D: N-by-K numpy array containing defined distance matrix (From N data points to K centeroids).
        alpha: Smoothing parameter
    Output:
        W: N-by-K numpy array containing weights
    """
    K = D.shape[1]
    W = np.zeros_like(D)
    J = np.sum(D * np.exp(-alpha * D), axis=1) / np.sum(np.exp(-alpha * D), axis=1)  # Objectives contributed by Q points individually
    for k in range(K):
        W[:, k] = np.exp(-alpha * D[:, k]) / np.sum(np.exp(-alpha * D), axis=1) * (1 - alpha * (D[:, k] - J))

    # Prevent all 0 membership because of numerical resolution
    zero_idx = np.where(np.sum(W, axis=1) == 0)[0]
    pos = np.argmin(D[zero_idx, :], axis=1)
    W[zero_idx, :] = 0
    W[zero_idx, pos] = 1

    return W
