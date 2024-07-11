import os

def create_project_structure(base_dir):
    folders = [
        'data/raw', 'data/processed', 'notebooks', 'src', 'tests', 'venv'
    ]
    
    files = {
        'README.md': '',
        'requirements.txt': '',
        '.gitignore': """# Virtual environment
venv/

# Python files
__pycache__/
*.pyc

# Jupyter Notebooks
.ipynb_checkpoints/

# Data files
data/raw/
data/processed/

# VSCode settings
.vscode/
""",
        'notebooks/EDA.ipynb': '',
        'src/__init__.py': '',
        'src/data_loader.py': """import cv2
import json
import os

def load_image(image_path):
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def load_json(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data

def save_processed_image(image, save_path):
    cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
""",
        'src/models.py': """import torch
import segmentation_models_pytorch as smp

def get_unet_model():
    return smp.Unet('resnet34', encoder_weights='imagenet', classes=1, activation='sigmoid')

def get_yolo_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model
""",
        'src/train.py': """import torch
from src.models import get_unet_model

def train_model(model, dataloader, num_epochs):
    for epoch in range(num_epochs):
        for images, masks in dataloader:
            # Training loop
            pass
    return model
""",
        'src/evaluate.py': """def evaluate_model(model, dataloader):
    for images, masks in dataloader:
        # Evaluation loop
        pass
    return metrics
""",
        'src/utils.py': """def visualize_image(image, title='Image'):
    import matplotlib.pyplot as plt
    plt.imshow(image)
    plt.title(title)
    plt.show()
""",
        'tests/__init__.py': '',
        'tests/test_data_loader.py': """import unittest
from src.data_loader import load_image, load_json

class TestDataLoader(unittest.TestCase):
    def test_load_image(self):
        image = load_image('path/to/sample.jpg')
        self.assertEqual(image.shape, (height, width, channels))

    def test_load_json(self):
        data = load_json('path/to/sample.json')
        self.assertIn('key', data)

if __name__ == '__main__':
    unittest.main()
"""
    }

    for folder in folders:
        os.makedirs(os.path.join(base_dir, folder), exist_ok=True)

    for file_path, content in files.items():
        with open(os.path.join(base_dir, file_path), 'w') as file:
            file.write(content)

if __name__ == '__main__':
    base_dir = './'
    create_project_structure(base_dir)
    print(f"Project structure created under: {base_dir}")
