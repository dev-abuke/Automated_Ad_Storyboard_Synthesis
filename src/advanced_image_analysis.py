import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
from PIL import Image

def load_data():
    image_features = np.load('image_features.npy')
    performance_data = pd.read_csv('processed_performance_data.csv')
    return image_features, performance_data

def investigate_clusters(image_features, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(image_features)
    
    # Find the closest points to each cluster center
    closest_points = []
    for i in range(n_clusters):
        cluster_points = image_features[cluster_labels == i]
        center = kmeans.cluster_centers_[i]
        distances = np.linalg.norm(cluster_points - center, axis=1)
        closest_point_idx = np.argmin(distances)
        closest_points.append(closest_point_idx)
    
    return cluster_labels, closest_points

def analyze_feature_importance(image_features):
    pca = PCA()
    pca.fit(image_features)
    
    # Calculate cumulative explained variance ratio
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    
    # Find number of components needed to explain 90% of variance
    n_components_90 = np.argmax(cumulative_variance_ratio >= 0.9) + 1
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('PCA: Cumulative Explained Variance Ratio')
    plt.axhline(y=0.9, color='r', linestyle='--')
    plt.axvline(x=n_components_90, color='r', linestyle='--')
    plt.text(n_components_90, 0.9, f'90% at {n_components_90} components', 
             verticalalignment='bottom', horizontalalignment='left')
    plt.savefig('pca_explained_variance.png')
    plt.close()
    
    return n_components_90
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
from PIL import Image

# ... (previous functions remain the same)

def correlate_with_performance(image_features, performance_data, cluster_labels):
    # Create a DataFrame with image features and cluster labels
    combined_data = pd.DataFrame(image_features)
    combined_data['Cluster'] = cluster_labels
    
    # Add a unique identifier to both dataframes
    combined_data['ID'] = range(len(combined_data))
    performance_data['ID'] = range(len(performance_data))
    
    # Merge the dataframes based on the ID
    merged_data = pd.merge(combined_data, performance_data[['ID', 'ER', 'CTR']], on='ID', how='inner')
    
    print(f"Number of samples with both image features and performance data: {len(merged_data)}")
    
    # Calculate average ER and CTR for each cluster
    cluster_performance = merged_data.groupby('Cluster')[['ER', 'CTR']].mean()
    
    # Visualize cluster performance
    cluster_performance.plot(kind='bar', figsize=(10, 6))
    plt.title('Average ER and CTR by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Rate')
    plt.legend(['ER', 'CTR'])
    plt.savefig('cluster_performance.png')
    plt.close()
    
    return cluster_performance, merged_data

def main():
    image_features, performance_data = load_data()
    
    # Investigate clusters
    cluster_labels, closest_points = investigate_clusters(image_features)
    print(f"Cluster sizes: {np.bincount(cluster_labels)}")
    print(f"Indices of closest points to cluster centers: {closest_points}")
    
    # Analyze feature importance
    n_components_90 = analyze_feature_importance(image_features)
    print(f"Number of components needed to explain 90% of variance: {n_components_90}")
    
    # Correlate with performance
    cluster_performance, merged_data = correlate_with_performance(image_features, performance_data, cluster_labels)
    print("Average ER and CTR by cluster:")
    print(cluster_performance)
    
    # Additional analysis on merged data
    print("\nCorrelation between image features and performance metrics:")
    correlation = merged_data.iloc[:, :-4].corrwith(merged_data['ER'])
    print("Top 5 features correlated with ER:")
    print(correlation.nlargest(5))
    print("\nTop 5 features correlated with CTR:")
    print(merged_data.iloc[:, :-4].corrwith(merged_data['CTR']).nlargest(5))
    
    print("\nAnalysis complete. Check the generated PNG files for visualizations.")

if __name__ == "__main__":
    main()

def main():
    image_features, performance_data = load_data()
    
    # Investigate clusters
    cluster_labels, closest_points = investigate_clusters(image_features)
    print(f"Cluster sizes: {np.bincount(cluster_labels)}")
    print(f"Indices of closest points to cluster centers: {closest_points}")
    
    # Analyze feature importance
    n_components_90 = analyze_feature_importance(image_features)
    print(f"Number of components needed to explain 90% of variance: {n_components_90}")
    
    # Correlate with performance
    cluster_performance = correlate_with_performance(image_features, performance_data, cluster_labels)
    print("Average ER and CTR by cluster:")
    print(cluster_performance)
    
    print("Analysis complete. Check the generated PNG files for visualizations.")

if __name__ == "__main__":
    main()