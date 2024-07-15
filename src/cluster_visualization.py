import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os

def load_image_data(assets_dir):
    image_data = []
    for root, dirs, files in os.walk(assets_dir):
        for file in files:
            # print(f"Loading images from directory: {root}")
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, file)
                image_data.append({'path': file_path, 'folder_path': root, 'array': None})
    return image_data

def load_image(file_path):
    try:
        with Image.open(file_path) as img:
            return np.array(img)
    except Exception as e:
        print(f"Error loading image {file_path}: {e}")
        return None

def visualize_cluster_representatives(indices, image_data, cluster_performance):
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Representative Images for Each Cluster", fontsize=16)

    for i, index in enumerate(indices):
        print(f"Index From Cluster Visualization is : {index} and len(image_data) : {len(image_data)}")
        if index < len(image_data):
            img_info = image_data[index]
            if img_info['array'] is None:
                # get image folder with the associated index with image_info['folder_path'] and load specific image
                img_folder = img_info['folder_path']
                # get the image with the image name '_preview.png' from the folder
                img_name = '_preview.png'

                preview_path = os.path.join(img_folder, img_name)

                print(f"The Preview image path is :: {preview_path} image folder path is :: {img_folder}")

                # img_info['array'] = load_image(img_info['path'])
                img_info['array'] = load_image(preview_path)
            
            if img_info['array'] is not None:
                row = i // 3
                col = i % 3
                axs[row, col].imshow(img_info['array'])
                axs[row, col].set_title(f"Cluster {i}\nER: {cluster_performance.loc[i, 'ER']:.3f}, CTR: {cluster_performance.loc[i, 'CTR']:.3f}")
                axs[row, col].axis('off')
        else:
            print(f"Index {index} is out of range for image_data")

    # Remove the empty subplot
    fig.delaxes(axs[1, 2])
    
    plt.tight_layout()
    plt.savefig('cluster_representatives.png')
    plt.close()

if __name__ == "__main__":
    assets_dir = 'data/adluido/Challenge_Data/Assets'
    indices = [280, 113, 2117, 692, 619]  # Indices of closest points to cluster centers
    
    image_data = load_image_data(assets_dir)
    print(f"Loaded {len(image_data)} images")

    # Correct cluster performance data
    cluster_performance = pd.read_csv('cluster_performance.csv')
    
    visualize_cluster_representatives(indices, image_data, cluster_performance)
    print("Cluster representative images have been saved as 'cluster_representatives.png'")