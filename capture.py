from pynput import keyboard
import time
import pandas as pd 

duration = 5
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

time.sleep(duration)
print(events)
data = pd.DataFrame(events)
data.to_csv('')