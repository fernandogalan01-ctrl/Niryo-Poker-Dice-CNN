# %%
import os
import sys

# FIX 1: Use 'r' before the string to make it a "raw string" so backslashes don't break
# FIX 2: Point to the actual folder, not the .code-workspace file
project_root = r"C:\Users\ferga\Documents\Niryo-Poker-Dice-CNN"
sys.path.append(project_root)

import torch
# Now this should work perfectly
from src.model import NiryoPokerCNN

def train_model():
    # Ensure the models directory exists
    os.makedirs("models", exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NiryoPokerCNN().to(device)
    
    print(f"Training started on {device}...")
    
    # --- PRO TIP: Add your data loading and training loop here ---
    
    # Example save
    torch.save(model.state_dict(), "models/poker_cnn.pth")
    print("Model saved to models/poker_cnn.pth")

if __name__ == "__main__":
    train_model()