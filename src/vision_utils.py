import cv2  
import numpy as np
def detect_dice(image):
    # Preprocesamiento
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Algoritmo Canny para bordes 
    edged = cv2.Canny(blurred, 50, 150)
    # Localización por contornos 
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dice_crops = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Filtrar por tamaño aproximado del dado
        if 500 < area < 5000: 
            x, y, w, h = cv2.boundingRect(cnt)
            roi = image[y:y+h, x:x+w]
            roi_resized = cv2.resize(roi, (64, 64)) 
            dice_crops.append(roi_resized)
    return dice_crops