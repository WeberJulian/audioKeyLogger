import numpy as np
import time
import pandas as pd 
import sounddevice as sd
from scipy.io.wavfile import write
from pynput import keyboard
import atexit

data = []
fs = 48000
events = []
correction = 0
sound = np.array([])
sleep = 1000 * 5

def save():
    global data
    global sound
    print("Writing output...")
    print(sound)
    data = pd.DataFrame(events)
    data["timestamp"] = data["timestamp"].apply(lambda a: (a - start)*1e-6)
    write('raw/output.wav', fs, sound)
    data.to_csv('raw/timestamps.csv')

atexit.register(save)

def on_press(key):
    global correction
    print(key)
    if hasattr(key, "char"):
        res = key.char
    else:
        res = str(key)
    events.append({'key': res, 'timestamp': time.time_ns()})
    print(f'key: {res} timestamp: {str(time.time_ns())} corrected_timestamp: {str(time.time_ns() - correction) }')


# listener.suppress()

def handleData(indata, frames, time, status):
    global sound
    sound = np.concatenate([data, indata.reshape((len(indata),))])

listener = keyboard.Listener(on_press=on_press)

with sd.InputStream(callback=handleData, samplerate=fs) as stream:
    listener.start()
    start = time.time_ns()
    while(1):
        first = True
        while (len(data) and time.time_ns()*1e-6  - data[-1]['timestamp']> sleep):
            
            if(first):
                print('sleeping...')
                first = False


        



print(len(data))