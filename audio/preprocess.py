import os 
import glob
import sys
import librosa
import soundfile as sf
from librosa.effects import trim
from librosa.util import normalize

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SAMPLE_RATE

def preprocess_audio(audio_array, sample_rate):
    """
    Trims and normalizes raw audio (1D numpy array).
    Returns: processed numpy array.
    """
    trimmed_audio, _ = trim(audio_array)
    normalised_audio = normalize(trimmed_audio)
    return normalised_audio

os.makedirs("data/processed_wakeword", exist_ok=True)
os.makedirs("data/processed_nonwakeword", exist_ok=True)

for file_path in glob.glob("data/raw_wakeword/*.wav"):
    audio_array, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    processed_audio = preprocess_audio(audio_array, SAMPLE_RATE)
    file_name = os.path.basename(file_path)
    output_path = os.path.join("data/processed_wakeword", file_name)
    sf.write(output_path, processed_audio, SAMPLE_RATE)

for file_path in glob.glob("data/raw_nonwakeword/*.wav"):
    audio_array, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    processed_audio = preprocess_audio(audio_array, SAMPLE_RATE)
    file_name = os.path.basename(file_path)
    output_path = os.path.join("data/processed_nonwakeword", file_name)
    sf.write(output_path, processed_audio, SAMPLE_RATE)
