import random
import numpy
import matplotlib.pyplot as plt
from read_datasets import *
import sys

numpy.set_printoptions(threshold=sys.maxsize, formatter={'float': lambda x: "{0:0.5f}".format(x)})


def pca(dataset, verbose=True):
    random.seed(27)
    n_samples = dataset.shape[0]

    if verbose:
        plt.scatter(dataset[:, 0], dataset[:, 1])
        plt.title(f'First Two Dimensions (Original)')
        plt.show()

    # Compute means and covariance matrix
    u = np.mean(dataset, axis=0)
    B = dataset - np.expand_dims(u, axis=0)
    cov = 1 / (n_samples - 1) * np.dot(B.T, B)
    if verbose:
        print('Covariance matrix')
        print(cov)

    eigenvalues,eigenvectors=np.linalg.eig(cov)

    if verbose:
        print('Eigenvectors')
        print(eigenvectors)
        print('Eigenvalues')
        print(eigenvalues)

    #Sort eigenvectors and eigenvalues
    sorted_index=np.argsort(-eigenvalues)
    eigenvalues=eigenvalues[sorted_index]
    eigenvectors=eigenvectors[:,sorted_index]
    if verbose:
        print('Sorted Eigenvectors')
        print(eigenvectors)
        print('Sorted Eigenvalues')
        print(eigenvalues)
    transformed_data=np.dot(B,eigenvectors)
    if verbose:
        plt.scatter(transformed_data[:, 0], transformed_data[:, 1])
        plt.title(f'First Two Dimensions (Transformed)')
        plt.show()
    recovered_data=np.dot(transformed_data,np.linalg.inv(eigenvectors))+np.expand_dims(u, axis=0)
    if verbose:
        plt.scatter(recovered_data[:, 0], recovered_data[:, 1])
        plt.title(f'First Two Dimensions (Recovered)')
        plt.show()
        print('Distance between original and recovered')
        print(np.linalg.norm(dataset-recovered_data))
    return transformed_data

if __name__ == '__main__':
    dataset = read_waveform()[0]
    pca(dataset)
