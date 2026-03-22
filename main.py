from pyniryo import NiryoRobot
from src.model import NiryoPokerCNN
from src.vision_utils import process_image_for_dice
from src.poker_logic import evaluate_hand
import torch
import cv2

# Configuración
robot_ip = "192.168.1.XX" # Cambia por la IP de tu Niryo
model = NiryoPokerCNN()
model.load_state_dict(torch.load("models/modelo_caras.pth"))
model.eval()

classes = ['9', '10', 'A', 'J', 'K', 'Q']

with NiryoRobot(robot_ip) as robot:
    print("Capturando imagen...")
    ret, frame = robot.get_img_compressed() # Captura desde el robot
    
    if ret:
        crops = process_image_for_dice(frame)
        results = []
        
        for img in crops:
            # Preparar para la CNN
            img_t = torch.tensor(img).permute(2,0,1).float().unsqueeze(0) / 255.0
            out = model(img_t)
            _, pred = torch.max(out, 1)
            results.append(classes[pred.item()])
        
        jugada = evaluate_hand(results)
        print(f"Jugada detectada: {jugada}")
        
        # Ejemplo de movimiento según jugada
        if jugada != "Nada":
            robot.pick_from_pose(0.2, 0.0, 0.1, 0, 0, 0)
            robot.place_from_pose(0.1, 0.2, 0.1, 0, 0, 0)