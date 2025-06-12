SAMPLE_RATE=16000 #Hz
CLIP_DURATION=2 #seconds
WAKEWORD_NAME="tanvi"
NON_WAKEWORD_NAME="random"
SAMPLE_RATE = 16000
N_MFCC = 13
MFCC_HOP_LENGTH = 160  # 10 ms hop if sample_rate = 16kHz

import numpy as np
import sounddevice as sd
def beep(freq=440, duration=0.3):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    tone = 0.5 * np.sin(2 * np.pi * freq * t)
    sd.play(tone, samplerate=SAMPLE_RATE)
    sd.wait()