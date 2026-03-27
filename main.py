import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

# --- 1. PATH CONFIGURATION ---
PROJECT_ROOT = r"C:\Users\ferga\Documents\Niryo-Poker-Dice-CNN"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from pyniryo import NiryoRobot
from src.model import NiryoPokerCNN
from src.vision_utils import process_image_for_dice
from src.poker_logic import evaluate_hand

def run_main():
    # --- 2. CONFIGURATION & MODEL LOADING ---
    robot_ip = "192.168.1.XX"  # <--- Update this to your Robot's IP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = NiryoPokerCNN().to(device)
    model_path = os.path.join(PROJECT_ROOT, "models", "modelo_caras.pth")
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found at {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    classes = ['9', '10', 'A', 'J', 'K', 'Q']

    # --- 3. TRANSFORMS (Must match training!) ---
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # --- 4. ROBOT EXECUTION ---
    try:
        with NiryoRobot(robot_ip) as robot:
            print("Connecting to Niryo...")
            robot.calibrate(robot.CALIBRATE_AUTO)
            
            print("Capturing image...")
            # Use compressed for speed, or raw for quality
            ret, frame = robot.get_img_compressed() 
            
            if ret:
                # Convert BGR (OpenCV) to RGB (PIL/Standard)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Extract individual dice crops
                crops = process_image_for_dice(frame_rgb)
                results = []
                
                print(f"Detected {len(crops)} dice. Predicting...")
                
                for img_np in crops:
                    # Convert numpy crop to PIL Image for transforms
                    img_pil = Image.fromarray(img_np)
                    img_t = transform(img_pil).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        out = model(img_t)
                        _, pred = torch.max(out, 1)
                        results.append(classes[pred.item()])
                
                print(f"Detected Faces: {results}")
                jugada = evaluate_hand(results)
                print(f"Result: {jugada}")
                
                # Example robot interaction logic
                if jugada != "Nada":
                    print("High value hand detected! Moving robot...")
                    # Add your specific move commands here
                    # robot.move_pose(...) 
            else:
                print("[ERROR] Could not capture image from robot.")

    except Exception as e:
        print(f"[ERROR] Connection failed: {e}")

if __name__ == "__main__":
    run_main()