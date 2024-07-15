import torch
import cv2
import matplotlib.pyplot as plt
import os

def load_yolo_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

def detect_objects(model, image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    
    # Convert the image to RGB (YOLOv5 expects RGB format)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # YOLOv5 expects the image in a numpy array format with shape (height, width, channels)
    # We don't need to resize as YOLOv5 handles different sizes internally
    results = model(image_rgb)
    
    return results

def plot_results(results, image_path, conf_threshold=0.5):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create figure and axes
    fig, ax = plt.subplots(1, figsize=(12, 8))
    
    # Display the image
    ax.imshow(image_rgb)
    
    # Define colors for different classes (you can expand this list)
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    
    # Plot each detection
    for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
        if conf >= conf_threshold:
            x1, y1, x2, y2 = map(int, xyxy)
            class_name = results.names[int(cls)]
            color = colors[int(cls) % len(colors)]
            
            # Draw bounding box
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor=color, linewidth=2)
            ax.add_patch(rect)
            
            # Add label
            ax.text(x1, y1-10, f'{class_name} {conf:.2f}', fontsize=8, color='white',
                    bbox=dict(facecolor=color, edgecolor='none', alpha=0.7))
    
    # Remove axes
    ax.axis('off')
    
    # Set title
    plt.title('Object Detection Results')
    
    # Adjust layout and display
    plt.tight_layout()
    
    # Create 'results' directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Save the plot as an image file
    output_path = os.path.join('results', 'object_detection_result.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    # Close the plot to free up memory
    plt.close(fig)
    
    print(f"Object detection result saved as '{output_path}'")
    return output_path