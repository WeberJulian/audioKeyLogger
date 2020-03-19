import matplotlib.pyplot as plt
from scipy.io import wavfile
import pandas as pd
import numpy as np
import uuid
import os

delay = 110 #ms
duration = 175 #ms
path = 'dataset/images/'

file_name = 'a8098c1a-f86e-11da-bd1a-00112444be1e'
# file_name = '10195770-6973-11ea-a3d7-acde48001122'

samplingFrequency, signalData = wavfile.read(f'raw/audio/{file_name}.wav')
data = pd.read_csv(f'raw/labels/{file_name}.csv', index_col=0)
ms2steps = lambda ms: int(ms*1e-3*samplingFrequency)
counter = 0
def generateImage(row):
    global counter
    line = row['timestamp']
    sample = signalData[ms2steps(line - delay):ms2steps(line - delay)+ms2steps(duration)]
    fig = plt.figure(frameon=False)
    fig.set_size_inches(2,2)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.specgram(sample,Fs=samplingFrequency,cmap='gray')
    image_name = str(uuid.uuid1())+'.png'
    plt.savefig(path+image_name, dpi=64)
    plt.close()
    counter +=1
    return image_name

data['timestamp'] = data.apply(lambda row: generateImage(row), axis=1)

if(os.path.isfile('dataset/timestamps.csv')):
    prev_data = pd.read_csv('dataset/timestamps.csv', index_col=0)
    data = pd.concat([prev_data, data])
    data.reset_index(inplace = True, drop=True) 

print(f'New images generated: {counter}')
print(f'Total: {len(data)}')

data.to_csv('dataset/timestamps.csv')