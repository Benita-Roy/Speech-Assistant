import os 
import glob
import sys
import librosa
import soundfile as sf
from librosa.effects import trim
from librosa.util import normalize
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SAMPLE_RATE

os.makedirs("data/processed_wakeword", exist_ok=True)

raw_file_dir="data/raw_wakeword"



for file_path in glob.glob("data/raw_wakeword/*.wav"):
    # Directly gives you full path of each .wav file
    audio_array, sr=librosa.load(file_path,sr=SAMPLE_RATE)
    trimmed_audio,_=trim(audio_array)
    normalised_audio=normalize(trimmed_audio)
    file_name=os.path.basename(file_path)
    output_path=os.path.join("data/processed_wakeword",file_name)
    sf.write(output_path,normalised_audio,SAMPLE_RATE)
    