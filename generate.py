import matplotlib.pyplot as plt
from scipy.io import wavfile
import pandas as pd
import numpy as np

delay = 85 #ms
duration = 75 #ms
path = 'dataset/images/'

samplingFrequency, signalData = wavfile.read('raw/output.wav')
data = pd.read_csv('raw/timestamps.csv', index_col=0)
ms2steps = lambda ms: int(ms*1e-3*samplingFrequency)

def generateImage(row):
    line = row['timestamp']
    sample = signalData[ms2steps(line - delay):ms2steps(line - delay)+ms2steps(duration)]
    fig = plt.figure(frameon=False)
    fig.set_size_inches(10,10)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.specgram(sample,Fs=samplingFrequency,cmap='gray')
    image_path = path+str(int(row['timestamp']))+'.png'
    plt.savefig(image_path, dpi=20)
    return image_path

data['timestamp'] = data.apply(lambda row: generateImage(row), axis=1)
print(data.head(5))
