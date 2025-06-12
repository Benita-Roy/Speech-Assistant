SAMPLE_RATE=16000 #Hz
CLIP_DURATION=2 #seconds
WAKEWORD_NAME="tanvi"
NON_WAKEWORD_NAME="random"
import numpy as np
import sounddevice as sd
def beep(freq=440, duration=0.3):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    tone = 0.5 * np.sin(2 * np.pi * freq * t)
    sd.play(tone, samplerate=SAMPLE_RATE)
    sd.wait()