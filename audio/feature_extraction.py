import librosa
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SAMPLE_RATE, N_MFCC, MFCC_HOP_LENGTH

def extract_mfcc(audio_array, sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC, hop_length=MFCC_HOP_LENGTH):
    audio_array = audio_array.flatten()  # Ensure 1D input
    mfcc = librosa.feature.mfcc(
        y=audio_array,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        hop_length=hop_length,
        n_fft=512,           # Avoid FFT > audio length
        win_length=400
    )
    return mfcc.T
