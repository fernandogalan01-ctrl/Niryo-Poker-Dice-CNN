import torch
import torch.nn as nn
class NiryoPokerCNN(nn.Module):
    def __init__(self):
        super(NiryoPokerCNN, self).__init__()
        # Definición de bloques Conv2d + BN + ReLU + MaxPool
        self.layer1 = self._make_layer(3, 32)   
        self.layer2 = self._make_layer(32, 64)  
        self.layer3 = self._make_layer(64, 128) 
        # Capas totalmente conectadas 
        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256), # Asumiendo que la imagen de entrada es de 64x64, después de 3 bloques de pooling se reduce a 8x8
            nn.ReLU(),# Activación ReLU
            nn.Dropout(0.5)# Para reducir el sobreajuste
        )
        self.fc2 = nn.Linear(256, 6) # 6 clases de salida (poker hand rankings)
    # Método auxiliar para crear bloques de capas
    def _make_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    # Definición del método forward
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return self.fc2(x)