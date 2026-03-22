import torch
from src.model import NiryoPokerCNN

def export_onnx():
    model = NiryoPokerCNN()
    model.load_state_dict(torch.load("models/modelo_caras.pth"))
    dummy_input = torch.randn(1, 3, 64, 64)
    torch.onnx.export(model, dummy_input, "models/poker_cnn.onnx")
    print("Modelo exportado a ONNX correctamente.")

if __name__ == "__main__":
    export_onnx()