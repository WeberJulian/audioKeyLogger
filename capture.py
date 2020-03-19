from pynput import keyboard
import numpy as np
import time
import pandas as pd 
import sounddevice as sd
from scipy.io.wavfile import write
import uuid

duration = 60
fs = 48000
events = []

def on_press(key):
    print(key)
    if hasattr(key, "char"):
        res = key.char
    else:
        res = str(key)
    events.append({'key': res, 'timestamp': time.time_ns()})
    print("key: "+res+" timestamp: "+str(time.time_ns()))

listener = keyboard.Listener(on_press=on_press)
listener.start()

print("Starting to listen for "+str(duration)+" seconds...")
sound = sd.rec(int(duration * fs), samplerate=fs, channels=1)
start = time.time_ns()
sd.wait()
listener.stop()

print("Writing output...")
name = str(uuid.uuid1())
data = pd.DataFrame(events)
data["timestamp"] = data["timestamp"].apply(lambda a: (a - start)*1e-6)
write(f'raw/audio/{name}.wav', fs, sound)
data.to_csv(f'raw/labels/{name}.csv')
print(f'Output name: {name}')