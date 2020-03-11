import torch
import torch.nn as nn

class ClassifierBaseCNN(nn.Module):
    def __init__(self, num_key, dropout=0.2):
        super(ClassifierBaseCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(dropout),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(dropout),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(dropout),
            nn.MaxPool2d(2)
        )
        self.dense1 = nn.Sequential(
            nn.Linear(32*16*16, 32*16*16),
            nn.LeakyReLU(),
            nn.BatchNorm1d(8192),
            nn.Dropout(dropout)
        )
        self.dense2 = nn.Sequential(
            nn.Linear(32*16*16, num_key)
        )
    
    def forward(self, image):
        x = self.conv1(image)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1,self.num_flat_features(x))
        x = self.dense1(x)
        x = self.dense2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class ClassifierSmallCNN(nn.Module):
    def __init__(self, num_key, dropout=0.2):
        super(ClassifierSmallCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 11, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(dropout),
            nn.MaxPool2d(4)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(dropout),
            nn.MaxPool2d(4)
        )
        self.dense1 = nn.Sequential(
            nn.Linear(32*16*16, 32*16*16),
            nn.LeakyReLU(),
            nn.BatchNorm1d(8192),
            nn.Dropout(dropout)
        )
        self.dense2 = nn.Sequential(
            nn.Linear(32*16*16, num_key)
        )
    
    def forward(self, image):
        x = self.conv1(image)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1,self.num_flat_features(x))
        x = self.dense1(x)
        x = self.dense2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
