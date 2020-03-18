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

samplingFrequency, signalData = wavfile.read(f'raw/audio/{file_name}.wav')
data = pd.read_csv(f'raw/labels/{file_name}.csv', index_col=0)
ms2steps = lambda ms: int(ms*1e-3*samplingFrequency)

def generateImage(row):
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
    return image_name

data['timestamp'] = data.apply(lambda row: generateImage(row), axis=1)

if(os.path.isfile('dataset/timestamps.csv')):
    prev_data = pd.read_csv(f'raw/labels/{file_name}.csv', index_col=0)
    data = pd.concat([prev_data, data])

print(len(data))
data.to_csv('dataset/timestamps.csv')