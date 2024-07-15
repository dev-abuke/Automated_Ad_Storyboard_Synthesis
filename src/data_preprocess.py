import os
import json
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from transformers import BertTokenizer, BertModel
import torch

import os
import json
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm

def preprocess_images(assets_dir, batch_size=100, max_images=10000):
    image_data = []
    image_files = []
    
    for root, dirs, files in os.walk(assets_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))
                if len(image_files) >= max_images:
                    break
        if len(image_files) >= max_images:
            break

    for i in tqdm(range(0, len(image_files), batch_size), desc="Processing images"):
        batch = image_files[i:i+batch_size]
        for file_path in batch:
            try:
                with Image.open(file_path) as img:
                    img = img.convert('RGB')  # Ensure image is in RGB format
                    img = img.resize((224, 224))  # Resize for consistency
                    img_array = np.array(img) / 255.0  # Normalize
                    image_data.append({
                        'file_path': file_path,
                        'array': img_array
                    })
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        # Save batch to avoid keeping all data in memory
        np.save(f'processed_image_data_batch_{i//batch_size}.npy', image_data)
        image_data = []  # Clear the list after saving

    return len(image_files)

if __name__ == "__main__":
    assets_dir = 'data/adluido/Challenge_Data/Assets'
    concepts_path = 'path/to/concepts.json'
    performance_data_path = 'path/to/performance_data.csv'
    
    # Preprocess images
    num_processed_images = preprocess_images(assets_dir, batch_size=100, max_images=10000)
    print(f"Processed {num_processed_images} images")
    
    # ... (rest of the main function remains the same)

def process_concepts(concepts_path):
    with open(concepts_path, 'r') as f:
        concepts = json.load(f)
    
    processed_concepts = []
    for concept in concepts:
        processed_concept = {
            'concept': concept['concept'],
            'explanation': concept['explanation'],
            'implementation': json.dumps(concept['implementation'])  # Convert dict to string
        }
        processed_concepts.append(processed_concept)
    
    return pd.DataFrame(processed_concepts)

def prepare_performance_data(performance_data_path):
    df = pd.read_csv(performance_data_path)
    
    # Normalize ER and CTR
    scaler = MinMaxScaler()
    df[['ER', 'CTR']] = scaler.fit_transform(df[['ER', 'CTR']])
    
    return df

def create_text_embeddings(df, column_name):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    embeddings = []
    for text in df[column_name]:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
    
    return np.array(embeddings)

if __name__ == "__main__":
    filepath = "/home/abubeker_shamil/Automated_Ad_Storyboard_Synthesis/data/adluido/Challenge_Data/performance_data.csv"
    assets_dir = '/home/abubeker_shamil/Automated_Ad_Storyboard_Synthesis/data/adluido/Challenge_Data/Assets'
    categories_path = '/home/abubeker_shamil/Automated_Ad_Storyboard_Synthesis/data/adluido/categories.txt'
    concepts_path = '/home/abubeker_shamil/Automated_Ad_Storyboard_Synthesis/data/adluido/concepts.json'
    performance_data_path = '/home/abubeker_shamil/Automated_Ad_Storyboard_Synthesis/data/adluido/Challenge_Data/performance_data.csv'
    
    # Preprocess images
    image_data = preprocess_images(assets_dir)
    print(f"Processed {len(image_data)} images")
    
    # Process concepts
    concepts_df = process_concepts(concepts_path)
    print(f"Processed {len(concepts_df)} concepts")
    
    # Create text embeddings for concept explanations
    concept_embeddings = create_text_embeddings(concepts_df, 'explanation')
    print(f"Created embeddings for concept explanations. Shape: {concept_embeddings.shape}")
    
    # Prepare performance data
    performance_df = prepare_performance_data(performance_data_path)
    print(f"Prepared performance data. Shape: {performance_df.shape}")
    
    # Save processed data
    np.save('processed_image_data.npy', image_data)
    concepts_df.to_csv('processed_concepts.csv', index=False)
    np.save('concept_embeddings.npy', concept_embeddings)
    performance_df.to_csv('processed_performance_data.csv', index=False)

    print("Data preprocessing completed and saved.")