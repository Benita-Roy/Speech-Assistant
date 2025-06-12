import os 
import sys
import time
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SAMPLE_RATE,CLIP_DURATION,WAKEWORD_NAME

def beep(freq=440, duration=0.3):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = 0.5 * np.sin(2 * np.pi * freq * t)
    sd.play(tone, samplerate=sample_rate)
    sd.wait()

duration=CLIP_DURATION
sample_rate=SAMPLE_RATE
samples=int(duration*sample_rate)
num_wakeword_samples=20

os.makedirs("data/raw_wakeword", exist_ok=True)

for i in range(num_wakeword_samples):
    print("Get ready to say the wakeword ")
    time.sleep(2)
    beep()
    recorded_audio=sd.rec(samples,samplerate=sample_rate,channels=1)
    sd.wait()
    filename=f"data/raw_wakeword/{WAKEWORD_NAME}_{i+1}.wav"
    try:
        write(filename,sample_rate,recorded_audio)
        print(f"Saved file{filename}")
    except Exception as e:
        print(f"Failed to save {filename} Error: {e}")