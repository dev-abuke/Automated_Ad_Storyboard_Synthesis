import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

def explore_assets_directory(assets_dir):
    print("Exploring Assets Directory:")
    for root, dirs, files in os.walk(assets_dir):
        level = root.replace(assets_dir, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f"{subindent}{f}")

def analyze_categories(categories_path):
    with open(categories_path, 'r') as file:
        categories = [line.strip() for line in file]
    print("\nCategories Analysis:")
    print(f"Total categories: {len(categories)}")
    print("Sample categories:", categories[:5])

def explore_concepts(concepts_path):
    with open(concepts_path, 'r') as file:
        concepts = json.load(file)
    print("\nConcepts Analysis:")
    print(f"Total concepts: {len(concepts)}")
    print("Sample concept structure:")
    print(json.dumps(concepts[0], indent=2))

def analyze_performance_data(performance_data_path):
    df = pd.read_csv(performance_data_path)
    print("\nPerformance Data Analysis:")
    print(df.info())
    print("\nSample data:")
    print(df.head())

    # Example analysis: Distribution of engagement rates
    plt.figure(figsize=(10, 6))
    df['ER'].hist(bins=20)
    plt.title('Distribution of Engagement Rates')
    plt.xlabel('Engagement Rate')
    plt.ylabel('Frequency')
    plt.show()

if __name__ == "__main__":
    assets_dir = '/home/abubeker_shamil/Automated_Ad_Storyboard_Synthesis/data/adluido/Challenge_Data/Assets'
    categories_path = 'data/adluido/categories.txt'
    concepts_path = 'data/adluido/concepts.json'
    performance_data_path = 'data/adluido/Challenge_Data/performance_data.csv'

    explore_assets_directory(assets_dir)
    analyze_categories(categories_path)
    explore_concepts(concepts_path)
    analyze_performance_data(performance_data_path)