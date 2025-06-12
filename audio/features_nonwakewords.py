import sys
import os
import librosa
import numpy as np
import glob
from librosa.feature import mfcc 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SAMPLE_RATE

os.makedirs("data/features_nonwakewords", exist_ok=True)


for file_path in glob.glob("data/processed_nonwakeword/*.wav"):
    audio_arr,sr=librosa.load(file_path,sr=SAMPLE_RATE)
    mfcc=librosa.feature.mfcc(y=audio_arr,sr=SAMPLE_RATE,n_mfcc=13)
    file_name=os.path.basename(file_path).replace(".wav",".npy")
    output_path=os.path.join("data/features_nonwakewords",file_name)
    np.save(output_path,mfcc.T)
    