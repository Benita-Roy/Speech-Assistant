import os
import sys
import time
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write

# Adjust path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SAMPLE_RATE, CLIP_DURATION
from config import beep  # You already defined this

# Words you'll be prompted to say (non-wakewords)
non_wakewords = [
    "hello", "okay", "bye", "yes", "no", "what", "can we",
    "sammy", "sandy", "tammy", "tanya", "tambi", "danvi", "start", "stop"
]

sample_rate = SAMPLE_RATE
duration = CLIP_DURATION
samples = int(duration * sample_rate)

output_dir = "data/raw_nonwakeword"
os.makedirs(output_dir, exist_ok=True)

print("\n Non-Wakeword Recording Starting...\n")

for word in non_wakewords:
    for i in range(3):  # Record each word 3 times
        print(f"Say: '{word}'  (Take {i+1})")
        time.sleep(1)
        beep()
        recorded_audio = sd.rec(samples, samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()

        filename = f"{output_dir}/{word}_{i+1}.wav"
        try:
            write(filename, sample_rate, recorded_audio)
            print(f" Saved: {filename}")
        except Exception as e:
            print(f" Failed to save {filename}: {e}")
        
        time.sleep(0.5)

print("\n✅ All non-wakeword recordings complete.\n")
