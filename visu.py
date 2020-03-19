import matplotlib.pyplot as plot
from scipy.io import wavfile
import pandas as pd
import numpy as np

delay = 110 #ms
duration = 175 #ms

file_name = '10195770-6973-11ea-a3d7-acde48001122'

samplingFrequency, signalData = wavfile.read(f'raw/audio/{file_name}.wav')
timestamps = list(pd.read_csv(f'raw/labels/{file_name}.csv')['timestamp'])
ms2steps = lambda ms: ms*1e-3*samplingFrequency

plot.subplot(211)
plot.title('Dataset')
plot.plot(signalData)
plot.xlabel('Sample')
plot.ylabel('Amplitude')
for line in timestamps:
    tmp = ms2steps(line - delay)
    plot.axvline(x=tmp, color='red')
    plot.axvline(x=tmp+ms2steps(duration), color='blue')

plot.subplot(212)
plot.specgram(signalData,Fs=samplingFrequency)
plot.xlabel('Time')
plot.ylabel('Frequency')

plot.show()