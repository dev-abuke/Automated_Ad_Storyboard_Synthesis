import torch
import segmentation_models_pytorch as smp

def get_unet_model():
    return smp.Unet('resnet34', encoder_weights='imagenet', classes=1, activation='sigmoid')

def get_yolo_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model
