import numpy as np
import os
from sklearn.decomposition import PCA
from skimage.feature import hog
from skimage import color
import cv2
from tqdm import tqdm

def load_image_batches(directory):
    image_data = []
    for file in os.listdir(directory):
        if file.startswith('processed_image_data_batch_') and file.endswith('.npy'):
            batch = np.load(os.path.join(directory, file), allow_pickle=True)
            image_data.extend(batch)
    return image_data

def extract_color_histogram(image, bins=32):
    try:
        # Ensure image is uint8 and has 3 channels
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        
        hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()
    except Exception as e:
        print(f"Error in extract_color_histogram: {e}")
        return np.zeros(bins * bins * bins)

def extract_hog_features(image):
    try:
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        gray_image = color.rgb2gray(image)
        features, _ = hog(gray_image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
        return features
    except Exception as e:
        print(f"Error in extract_hog_features: {e}")
        return np.zeros(64)  # Adjust this based on your expected HOG feature size

def extract_features(image_data):
    features = []
    for item in tqdm(image_data, desc="Extracting features"):
        try:
            image = item['array']
            color_hist = extract_color_histogram(image)
            hog_feat = extract_hog_features(image)
            combined_features = np.concatenate([color_hist, hog_feat])
            features.append(combined_features)
        except Exception as e:
            print(f"Error processing image {item.get('file_path', 'unknown')}: {e}")
            # Append a zero vector if feature extraction fails
            features.append(np.zeros(32*32*32 + 64))
    return np.array(features)

def apply_pca(features, n_components=100):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(features)

if __name__ == "__main__":
    # Load processed image data
    image_data = load_image_batches('.')
    print(f"Loaded {len(image_data)} processed images")

    # Extract features
    print("Extracting features...")
    features = extract_features(image_data)
    print(f"Extracted features shape: {features.shape}")

    # Apply PCA
    print("Applying PCA...")
    pca_features = apply_pca(features)
    print(f"PCA features shape: {pca_features.shape}")

    # Save features
    np.save('image_features.npy', pca_features)
    print("Saved image features to image_features.npy")