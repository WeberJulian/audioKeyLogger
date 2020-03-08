import os
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from skimage import io, transform
import matplotlib.pyplot as plt

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
    def __init__(self, num_key, dropout=0.2):
        super(ClassifierCNN, self).__init__()
        self.conv1 = nn.Sequential(
            #nn.Conv2d(1, 16, (5,3), padding=(2,1)),
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
            nn.Conv2d(32, 64, 3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(dropout),
            nn.MaxPool2d(2)
        )
        # self.conv4 = nn.Sequential(
        #     nn.Conv2d(32, 64, 3, padding=1),
        #     nn.LeakyReLU(),
        #     nn.BatchNorm2d(64),
        #     nn.Dropout2d(dropout),
        #     nn.MaxPool2d(2)
        # ) 
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

train_length = int(len(dataset)*0.8)
validation_length = int(len(dataset)*0.1)
testing_length = len(dataset) - (validation_length + train_length)

lengths = [train_length, validation_length, testing_length]
train_set, validation, test_set = torch.utils.data.random_split(dataset, lengths)
trainloader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True, num_workers=0)
validationloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = ClassifierCNN(num_key=dataset.getNumKey(), dropout=0.2)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)

acc_val = []
acc_train = []
train_loss = []

for epoch in range(10):
    print('epoch -- '+str(epoch))
    total = 0
    correct = 0
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()
        if(inputs.size()[0] != 1):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels.cpu()).sum()
        running_loss += loss.item()
        if i % 10 == 9:
            acc_train.append(100.00 * correct.numpy() / total)
            model.eval()
            total = 0
            correct = 0
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted.cpu() == labels.cpu()).sum()
            validation_accuracy = 100.00 * correct.numpy() / total
            acc_val.append(validation_accuracy)
            print('%d loss: %.4f train_accuracy: %.3f val_accuracy: %.3f' % (i + 1, running_loss / 10, acc_train[-1], acc_val[-1]))
            running_loss = 0.0
            total = 0
            correct = 0
            model.train()

print('Finished Training')

plt.title("Accuracy")
plt.plot(acc_train, c='r', label='training')
plt.plot(acc_val, c='b', label='testing')
plt.legend()
plt.grid(b=None, which='both', axis='y')
plt.show()