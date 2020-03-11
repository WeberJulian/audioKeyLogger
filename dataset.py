import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io

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