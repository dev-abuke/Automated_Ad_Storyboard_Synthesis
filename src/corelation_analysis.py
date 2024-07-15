import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    image_features = np.load('image_features.npy')
    performance_data = pd.read_csv('processed_performance_data.csv')
    return image_features, performance_data

def correlate_features_with_performance(image_features, performance_data):
    # Ensure we only use data points for which we have both features and performance metrics
    n_samples = min(len(image_features), len(performance_data))
    image_features = image_features[:n_samples]
    performance_data = performance_data.iloc[:n_samples]

    # Combine features and performance data
    combined_data = pd.DataFrame(image_features)
    combined_data['ER'] = performance_data['ER'].values
    combined_data['CTR'] = performance_data['CTR'].values
    
    # Calculate correlation
    correlation_er = combined_data.corr()['ER'].sort_values(ascending=False)
    correlation_ctr = combined_data.corr()['CTR'].sort_values(ascending=False)
    
    return correlation_er, correlation_ctr

def visualize_top_features(correlation_er, correlation_ctr):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    correlation_er[:10].plot(kind='bar')
    plt.title('Top 10 Features Correlated with ER')
    plt.subplot(1, 2, 2)
    correlation_ctr[:10].plot(kind='bar')
    plt.title('Top 10 Features Correlated with CTR')
    plt.tight_layout()
    plt.savefig('top_correlated_features.png')
    plt.close()

def analyze_performance_distribution(performance_data):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(performance_data['ER'], kde=True)
    plt.title('Distribution of Engagement Rate')
    plt.subplot(1, 2, 2)
    sns.histplot(performance_data['CTR'], kde=True)
    plt.title('Distribution of Click-Through Rate')
    plt.tight_layout()
    plt.savefig('performance_distribution.png')
    plt.close()

def main():
    image_features, performance_data = load_data()
    
    print(f"Number of image features: {len(image_features)}")
    print(f"Number of performance data points: {len(performance_data)}")
    
    correlation_er, correlation_ctr = correlate_features_with_performance(image_features, performance_data)
    visualize_top_features(correlation_er, correlation_ctr)
    
    analyze_performance_distribution(performance_data)
    
    print("Top 5 features correlated with ER:")
    print(correlation_er[:5])
    print("\nTop 5 features correlated with CTR:")
    print(correlation_ctr[:5])
    
    print("\nAnalysis complete. Check the generated PNG files for visualizations.")

if __name__ == "__main__":
    main()