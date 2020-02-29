from pynput import keyboard
import time
import pandas as pd 
import sounddevice as sd
from scipy.io.wavfile import write

duration = 5
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
sd.wait()

listener.stop()
print("Playback...")
sd.play(sound, fs)
sd.wait()

print("Writing output...")
data = pd.DataFrame(events)
data.to_csv('dataset/timestamp.csv')
write('dataset/output.wav', fs, sound)