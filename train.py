import os
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from skimage import io, transform

path = 'dataset/'

class AudioKeyLoggerDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.timestamps = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        self.classes = self.timestamps['key'].unique()
        self.labels = dict({})
        for i in range(len(self.classes)):
            self.labels.update({self.classes[i]: i})

    def __len__(self):
        return len(self.timestamps)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.image_dir, self.timestamps.iloc[idx, 2])
        image = io.imread(img_name, as_gray=True)
        key = self.timestamps.iloc[idx, 1]
        return { 'image': torch.tensor(image).unsqueeze(0), 'key': self.labels.get(key)}
    
    def getNumKey(self):
        return len(self.classes)

class ClassifierCNN(nn.Module):
    def __init__(self, num_key):
        super(ClassifierCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2)
        )
        self.dense1 = nn.Sequential(
            nn.Linear(64*16*16, 32*16*16),
            nn.LeakyReLU(),
            nn.BatchNorm1d(8192)
        )
        self.dense2 = nn.Sequential(
            nn.Linear(32*16*16, num_key)
        )
    
    def forward(self, image):
        x = self.conv1(image)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(1,-1)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

dataset = AudioKeyLoggerDataset(os.path.join(path, 'timestamps.csv'), os.path.join(path, 'images'))

lengths = [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)]
train_set, test_set = torch.utils.data.random_split(dataset, lengths)
testloader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(test_set, batch_size=16, shuffle=True, num_workers=2)

model = ClassifierCNN(num_key=dataset.getNumKey())

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam()

for epoch in range(5):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')