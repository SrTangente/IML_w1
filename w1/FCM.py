import numpy as np
from numpy.linalg import norm


def FCM(dataset, k, m, it=100, tol=0.01, crisp=True):
    '''
    Inputs:
        dataset: numpy array of (sample,feature)
        k: int value for number of clusters
        m: float value for U exponent
        it: int value for max number of iterations
        tol: float value for minimum improvement of performance before stopping
        crisp: boolean value. If true return crisp clusters else return memberships.
    Outputs:
        c: numpy array of centroids.
        m: membership or crisp clusters.
    '''
    n_feat = dataset.shape[1]
    n_samples = dataset.shape[0]

    # Random init and normalize universe matrix
    u_0 = np.random.normal(0, 1, [k, n_samples])
    u_0 = u_0 / np.sum(u_0, axis=1, keepdims=True)

    # Stack dataset k times to vectorize difference norm calculations
    stacked = np.stack([dataset] * k, axis=1)

    for i in range(it):
        # Get new centroids
        v = np.dot(u_0 ** m, dataset) / np.sum(u_0 ** m, axis=1, keepdims=True)

        # Take norms
        norms = norm(stacked - v, axis=2)

        # Update universe
        u = 1 / ((norms ** (2 / (m - 1)) * np.sum(1/norms ** (2 / (m - 1)), axis=1,keepdims=True)))

        u = np.transpose(u)
        # Break if universe does not change enough
        if norm(u_0 - u) < tol:
            print('Change in universe below tolerance. Stopping.')
            break
        u_0 = u.copy()
    if not crisp:
        return np.concatenate([dataset, np.transpose(u)],axis=1)
    else:
        return np.concatenate([dataset, np.reshape(np.argmax(u, axis=0),[n_samples,1])],axis=1)
