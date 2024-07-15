import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

def load_features(file_path):
    return np.load(file_path)

def visualize_pca_variance(features):
    pca_variance = np.var(features, axis=0)
    cumulative_variance = np.cumsum(pca_variance) / np.sum(pca_variance)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Cumulative Explained Variance Ratio vs. Number of PCA Components')
    plt.grid(True)
    plt.savefig('pca_variance.png')
    plt.show()
    plt.close()

def apply_tsne(features):
    tsne = TSNE(n_components=2, random_state=42)
    return tsne.fit_transform(features)

def visualize_tsne(tsne_results):
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.5)
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.title('t-SNE visualization of image features')
    plt.savefig('tsne_visualization.png')
    plt.close()

def apply_kmeans(features, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    return kmeans.fit_predict(features)

def visualize_kmeans(tsne_results, cluster_labels):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=cluster_labels, cmap='viridis', alpha=0.5)
    plt.colorbar(scatter)
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.title('K-means clustering of image features (t-SNE visualization)')
    plt.savefig('kmeans_clustering.png')
    plt.close()

if __name__ == "__main__":
    # Load the features
    features = load_features('image_features.npy')
    print(f"Loaded features with shape: {features.shape}")

    # Visualize PCA variance
    visualize_pca_variance(features)
    print("PCA variance visualization saved as 'pca_variance.png'")

    # Apply t-SNE
    tsne_results = apply_tsne(features)
    visualize_tsne(tsne_results)
    print("t-SNE visualization saved as 'tsne_visualization.png'")

    # Apply K-means clustering
    cluster_labels = apply_kmeans(features)
    visualize_kmeans(tsne_results, cluster_labels)
    print("K-means clustering visualization saved as 'kmeans_clustering.png'")

    print("Analysis complete. Check the generated PNG files for visualizations.")