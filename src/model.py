import torch
import torch.nn as nn

#class PokemonClassifier(nn.Module):
#
#    def __init__(self, num_classes):
#        super().__init__()
#
#        #Feature Extractor
#        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
#        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
#        self.conv3 = nn.Conv2d(in_channels= 32, out_channels= 64, kernel_size=3, padding=1)
#        self.bn1 = nn.BatchNorm2d(16)
#        self.bn2 = nn.BatchNorm2d(32)
#        self.bn3 = nn.BatchNorm2d(64)

#        self.relu = nn.ReLU()
#        self.dropout = nn.Dropout(0.5)
#        self.pool = nn.MaxPool2d(kernel_size=2)
#        self.flatten = nn.Flatten()

        #Classifier
#        self.fc1 = nn.Linear(64 * 16 * 16 , 512)
#        self.fc2 = nn.Linear(512, num_classes)

#    def forward(self, x):
        #feature extraction
#        x = self.conv1(x)
#        x = self.bn1(x)
#        x = self.relu(x)
#        x = self.pool(x)

#        x = self.conv2(x)
#        x = self.bn2(x)
#        x = self.relu(x)
#        x = self.pool(x)

#        x = self.conv3(x)
#        x = self.bn3(x)
#        x = self.relu(x)
#        x = self.pool(x)

        #classification
#        x = self.flatten(x)
#        x = self.fc1(x)
#        x = self.relu(x)
#        x = self.dropout(x)
#        x = self.fc2(x)

#        return x
    
#model = PokemonClassifier(num_classes=len(dataset_classes))
#dummy = torch.randn(32, 3, 128, 128)
#output = model(dummy)
#print(output.shape)
print("="*50)
from torchvision.models import (resnet18, ResNet18_Weights)
print("loading model")
def get_resnet18(num_classes):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.layer4.parameters():
        param.requires_grad = True
    
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
