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
        return (torch.from_numpy(image).unsqueeze(0).float(), self.labels.get(key))
    
    def getNumKey(self):
        return len(self.classes)

    def getKeyFromClass(self):
        return None

class ClassifierCNN(nn.Module):
    def __init__(self, num_key, dropout=0.05):
        super(ClassifierCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, (5,3), padding=(2,1)),
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
            nn.Conv2d(32, 64, 3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(dropout),
            nn.MaxPool2d(2)
        )
        self.dense1 = nn.Sequential(
            nn.Linear(64*16*16, 32*16*16),
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

dataset = AudioKeyLoggerDataset(os.path.join(path, 'timestamps.csv'), os.path.join(path, 'images'))

lengths = [int(len(dataset)*0.9), len(dataset)-int(len(dataset)*0.9)]
train_set, test_set = torch.utils.data.random_split(dataset, lengths)
trainloader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=0)

model = ClassifierCNN(num_key=dataset.getNumKey())

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=5e-5)

for epoch in range(10):  # loop over the dataset multiple times
    print('epoch -- '+str(epoch))

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        if(inputs.size()[0] != 1):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:    # print every 2000 mini-batches
            model.eval()
            total = 0
            correct = 0
            for inputs, labels in testloader:
                output = model.forward(inputs)
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum() #sommation des éléments prédits.
                test_accuracy = 100.00 * correct.numpy() / total
            model.train()
            print('[%d, %5d] loss: %.6f accuracy: %.3f' % (epoch + 1, i + 1, running_loss / 2000, test_accuracy))
            running_loss = 0.0

print('Finished Training')