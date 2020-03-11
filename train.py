import os
import torch
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from model import ClassifierBaseCNN, ClassifierSmallCNN
from dataset import AudioKeyLoggerDataset


dataset_path = 'dataset/'

dataset = AudioKeyLoggerDataset(os.path.join(dataset_path, 'timestamps.csv'), os.path.join(dataset_path, 'images'))

train_length = int(len(dataset)*0.9)
validation_length = int(len(dataset)*0.1)
testing_length = len(dataset) - (validation_length + train_length)

lengths = [train_length, validation_length, testing_length]
train_set, validation_set, test_set = torch.utils.data.random_split(dataset, lengths)
trainloader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True, num_workers=0)
validationloader = torch.utils.data.DataLoader(validation_set, batch_size=16, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train():
    model = ClassifierBaseCNN(num_key=dataset.getNumKey(), dropout=0.1)
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)

    acc_val = []
    acc_train = []
    train_loss = []
    avg_epoch = []

    epoch = 0
    len_epoch = 0
    seuil = 0.5
    patience = 4
    learned = False

    while not learned:
        #print('epoch -- '+str(epoch))
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
                if epoch == 0:
                    len_epoch = len(acc_train)
                model.eval()
                total = 0
                correct = 0
                for inputs, labels in validationloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted.cpu() == labels.cpu()).sum()
                validation_accuracy = 100.00 * correct.numpy() / total
                acc_val.append(validation_accuracy)
                #print('%d loss: %.4f train_accuracy: %.3f val_accuracy: %.3f' % (i + 1, running_loss / 10, acc_train[-1], acc_val[-1]))
                running_loss = 0.0
                total = 0
                correct = 0
                model.train()

        avg_epoch.append(sum(acc_val[-len_epoch:])/len_epoch)
        #print('Average validation accurarcy: '+ str(avg_epoch[-1]))
        if(epoch+1 > patience):
            if(avg_epoch[-1] - min(avg_epoch[-(patience+1):]) <= seuil):
                learned = True
        epoch+=1

    #print('Finished Training')
    print('Maximum average accuracy: ' + str(max(avg_epoch)))
    return max(avg_epoch)

def benchmark(n):
    results = []
    for i in range(n):
        results.append(train())
        print(str(i) + 'eme moyenne: ' + str(sum(results)/len(results)))

benchmark(10)
# plt.title("Accuracy")
# plt.plot(acc_train, c='r', label='training')
# plt.plot(acc_val, c='b', label='testing')
# plt.legend()
# plt.grid(b=None, which='both', axis='y')
# plt.show()