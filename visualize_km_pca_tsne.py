from k_means import kmeans
import read_datasets
import evaluate
from sklearn import decomposition, manifold
from matplotlib import pyplot as plt
from pca import pca



def visualize_km(data, classes, k=3, num_components=5, perplexity=30, alpha=0.5):


    # Compute k-means without dimensionality reduction
    labels_without = kmeans(data, k)[:, -1]

    # Compute k-means with PCA
    pca_data = pca(data, False)[:, :num_components]
    labels_pca = kmeans(pca_data, k)[:, -1]

    # Project data to t-SNE
    tsne = manifold.TSNE(2, perplexity=perplexity)
    tsne_data = tsne.fit_transform(data)

    fig1, axes1 = plt.subplots(3, 2, figsize=(8, 8))
    plt.subplots_adjust(top=0.961, bottom=0.062, left=0.1, right=0.991, hspace=0.4, wspace=0.3)
    # Plot original labels with 2 principal components (PCA)
    axes1[0, 0].set_title('(a) Original labels (PCA)')
    axes1[0, 0].set_xlabel('1st principal component')
    axes1[0, 0].set_ylabel('2nd principal component')
    axes1[0, 0].scatter(pca_data[:, 0], pca_data[:, 1], c=classes, alpha=alpha, cmap="rainbow")

    # Plot original labels with 2 principal components (t-SNE)
    axes1[0, 1].set_title('(b) Original labels (t-SNE)')
    axes1[0, 1].set_xlabel('1st principal component')
    axes1[0, 1].set_ylabel('2nd principal component')
    axes1[0, 1].scatter(tsne_data[:, 0], tsne_data[:, 1], c=classes, alpha=alpha, cmap="rainbow")

    # Plot k-means without dim red (PCA)
    axes1[1, 0].set_title('(c) K-Means with original data (PCA)')
    axes1[1, 0].set_xlabel('1st principal component')
    axes1[1, 0].set_ylabel('2nd principal component')
    axes1[1, 0].scatter(pca_data[:, 0], pca_data[:, 1], c=labels_without, alpha=alpha, cmap="rainbow")

    # Plot k-means without dim red (t-SNE)
    axes1[1, 1].set_title('(d) K-Means with original data (t-SNE)')
    axes1[1, 1].set_xlabel('1st principal component')
    axes1[1, 1].set_ylabel('2nd principal component')
    axes1[1, 1].scatter(tsne_data[:, 0], tsne_data[:, 1], c=labels_without, alpha=alpha, cmap="rainbow")

    # Plot k-means after PCA (PCA)
    axes1[2, 0].set_title('(e) K-Means after PCA (PCA)')
    axes1[2, 0].set_xlabel('1st principal component')
    axes1[2, 0].set_ylabel('2nd principal component')
    axes1[2, 0].scatter(pca_data[:, 0], pca_data[:, 1], c=labels_pca, alpha=alpha, cmap="rainbow")

    # Plot k-means after PCA (t-SNE)
    axes1[2, 1].set_title('(f) K-Means after PCA (t-SNE)')
    axes1[2, 1].set_xlabel('1st principal component')
    axes1[2, 1].set_ylabel('2nd principal component')
    axes1[2, 1].scatter(tsne_data[:, 0], tsne_data[:, 1], c=labels_pca, alpha=alpha, cmap="rainbow")

    plt.show()
