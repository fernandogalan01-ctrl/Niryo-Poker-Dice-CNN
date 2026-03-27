
import os
import sys

# --- 1. PATH CONFIGURATION ---
# Use a raw string (r"") to prevent Windows backslash errors (e.g., \U, \n)
PROJECT_ROOT = r"C:\Users\ferga\Documents\Niryo-Poker-Dice-CNN"

# Add the project root to sys.path so 'import src' works
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# --- 2. IMPORTS ---
import torch
import numpy as np  
# --- 1. PATH CONFIGURATION ---
# Use a raw string (r"") to prevent Windows backslash errors (e.g., \U, \n)
PROJECT_ROOT = r"C:\Users\ferga\Documents\Niryo-Poker-Dice-CNN"

# Add the project root to sys.path so 'import src' works
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Also add the current directory for safety
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# --- 2. IMPORTS ---
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Import your custom model from the src folder
try:
    from src.model import NiryoPokerCNN
except ModuleNotFoundError:
    print("\n[ERROR] Could not find 'src.model'.")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)

def run_evaluation():
    # --- 3. CONFIGURATION & MODEL LOADING ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = NiryoPokerCNN().to(device)
    
    # Ensure the path to the weight file is correct
    model_path = os.path.join(PROJECT_ROOT, "models", "modelo_caras.pth")
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Model weights not found at: {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully.")

    # --- 4. IMAGE TRANSFORMATIONS ---
    # Note: These must match the transformations used during training!
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # --- 5. DATA LOADING ---
    test_path = os.path.join(PROJECT_ROOT, "data", "test")
    if not os.path.exists(test_path):
        print(f"[ERROR] Test data directory not found at: {test_path}")
        return

    test_data = datasets.ImageFolder(test_path, transform=transform)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    
    classes = test_data.classes
    all_preds = []
    all_labels = []

    # --- 6. INFERENCE ---
    print(f"Evaluating model on {len(test_data)} images...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- 7. QUANTITATIVE RESULTS ---
    print("\n" + "="*30)
    print("   QUANTITATIVE RESULTS")
    print("="*30)
    print(classification_report(all_labels, all_preds, target_names=classes))
    
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f"Global Weighted F1-Score: {f1:.4f}")

    # --- 8. CONFUSION MATRIX ---
    # Ensure the directory exists before saving the plot
    output_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(output_dir, exist_ok=True)

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.xlabel('Prediction')
    plt.ylabel('Ground Truth')
    plt.title('Confusion Matrix - Niryo Poker Dice')
    
    save_path = os.path.join(output_dir, "matriz_confusion.png")
    plt.savefig(save_path)
    print(f"\nConfusion matrix saved to: {save_path}")
    
    # Display the plot
    plt.show()

if __name__ == "__main__":
    run_evaluation()