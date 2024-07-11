import torch
from src.models import get_unet_model

def train_model(model, dataloader, num_epochs):
    for epoch in range(num_epochs):
        for images, masks in dataloader:
            # Training loop
            pass
    return model
