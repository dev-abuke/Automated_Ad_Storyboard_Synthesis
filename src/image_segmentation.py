import segmentation_models_pytorch as smp
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_unet_model():
    model = smp.Unet('resnet34', encoder_weights='imagenet', classes=1, activation='sigmoid')
    return model

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256))
    image = image / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
    return image

def segment_image(model, image_tensor):
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
    return output.squeeze().numpy()

def plot_segmentation(image_path, segmentation):
    image = cv2.imread(image_path)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(segmentation, cmap='gray')
    plt.title('Segmentation')
    plt.axis('off')

    plt.show()