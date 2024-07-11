import cv2
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
