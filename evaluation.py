import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from src.model import NiryoPokerCNN
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def run_evaluation():
    # 1. Configuración y Carga del Modelo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NiryoPokerCNN().to(device)
    model.load_state_dict(torch.load("models/modelo_caras.pth", map_location=device))
    model.eval()

    # 2. Transformaciones para las imágenes de test (deben coincidir con el entrenamiento)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 3. Cargar Dataset de Test (Asegúrate de tener una carpeta 'data/test')
    # Las subcarpetas deben llamarse: 9, 10, J, Q, K, A
    test_data = datasets.ImageFolder("data/test", transform=transform)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    
    classes = test_data.classes
    all_preds = []
    all_labels = []

    # 4. Inferencia
    print("Evaluando modelo...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 5. Cálculo de Métricas (Para la memoria técnica)
    print("\n--- RESULTADOS CUANTITATIVOS ---")
    print(classification_report(all_labels, all_preds, target_names=classes))
    
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f"F1-Score Global: {f1:.4f}")

    # 6. Matriz de Confusión (Generar el gráfico)
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.xlabel('Predicción')
    plt.ylabel('Realidad')
    plt.title('Matriz de Confusión - Dados de Póker')
    
    # Guardar la imagen para la memoria técnica
    plt.savefig("models/matriz_confusion.png")
    print("\nMatriz de confusión guardada en 'models/matriz_confusion.png'")
    plt.show()

if __name__ == "__main__":
    run_evaluation()