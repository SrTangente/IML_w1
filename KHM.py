import numpy as np
from numpy.linalg import norm


def KHM(dataset, k, p, it, tol, crisp=True):
    '''
    Inputs:
        dataset: numpy array of (sample,feature)
        k: int value for number of clusters
        p: float value for Lp norm used
        it: int value for max number of iterations
        tol: float value for minimum improvement of performance before stopping
        crisp: boolean value. If true return crisp clusters else return memberships.
    Outputs:
        c: numpy array of centroids.
        m: membership or crisp clusters.
    '''
    n_feat = dataset.shape[1]
    n_samples = dataset.shape[0]

    # Original paper recommends using random samples of the dataset to initialize centroids
    c_0 = np.arange(n_samples)
    np.random.shuffle(c_0)
    c_0 = dataset[c_0[:k]]

    # Init performance value to infty
    perf_0 = np.infty


    for i in range(it):
        print(f'Iteration {i}')
        #Get difference norms of each sample to each centroid
        stacked = np.stack([dataset] * k, axis=1)
        norms = np.maximum(norm(stacked - c_0, p, axis=2), np.finfo(float).eps)

        #Get membership and weight values
        m = norms ** (-p - 2) / np.sum(norms ** (-p - 2), axis=1, keepdims=True) #(sample,cluster)
        w = np.sum(norms ** (-p - 2), axis=1) / np.sum(norms ** (-p), axis=1) ** 2 #(sample)

        #Get new centroids. Note we use matrix product above
        c = np.dot(np.transpose(m * np.expand_dims(w, axis=1)), dataset) / np.expand_dims(
            np.sum(m * np.expand_dims(w, axis=1), axis=0), axis=1)

        #Get new performance value
        perf = np.sum(k / np.sum(1 / norm(stacked - c_0, p,axis=1) ** p), axis=0)

        #Break if performance does not improve enough
        if perf_0 - perf < tol:
            print('Change in performance below tolerance. Stopping.')
            break
        perf_0 = perf.copy()
        c_0 = c.copy()
    if not crisp:
        return c, m
    else:
        return c, np.argmax(m, axis=1)
