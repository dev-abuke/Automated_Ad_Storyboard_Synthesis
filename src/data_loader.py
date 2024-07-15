import os
import cv2
import json
import pandas as pd

def load_image(image_path):
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def load_json(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data

def load_txt(txt_path):
    with open(txt_path, 'r') as file:
        data = file.readlines()
    return [line.strip() for line in data]

def load_csv(csv_path):
    return pd.read_csv(csv_path)

def save_processed_image(image, save_path):
    cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
