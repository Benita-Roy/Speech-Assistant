from model.model_wakeword_final import WakeWordModel 
import torch
import torchaudio
import sounddevice as sd 
import numpy as np
from audio.preprocess import preprocess_audio
from audio.feature_extraction import extract_mfcc
from config import SAMPLE_RATE, beep

# --- Constants ---
duration = 2
EXPECTED_SHAPE = (63, 13)  # the input shape used during training

# --- Model Setup ---
model = WakeWordModel(input_shape=EXPECTED_SHAPE)

# Load state dict safely (avoid warning)
state_dict = torch.load("wakeword_model.pth", weights_only=True)
model.load_state_dict(state_dict)
model.eval()

# --- Real-time Inference Loop ---
# --- Real-time Inference Loop ---
THRESHOLD = 0.95  # You can tune this for fewer false positives

while True:
    # Record audio
    audio = sd.rec(int(SAMPLE_RATE * duration), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()

    # Preprocess and extract MFCC
    audio = preprocess_audio(audio, SAMPLE_RATE)
    mfcc = extract_mfcc(audio, SAMPLE_RATE)

    # Pad or crop to match training shape
    padded_mfcc = np.zeros(EXPECTED_SHAPE)
    t_min = min(EXPECTED_SHAPE[0], mfcc.shape[0])
    f_min = min(EXPECTED_SHAPE[1], mfcc.shape[1])
    padded_mfcc[:t_min, :f_min] = mfcc[:t_min, :f_min]

    # Convert to tensor
    feature_tensor = torch.tensor(padded_mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Predict with confidence
    with torch.no_grad():
        output = model(feature_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence = probabilities[0, 1].item()  # Confidence for class 1 (wake word)

    # Act only if above threshold
    if confidence > THRESHOLD:
        beep()
        print(f"Wake word detected! (confidence: {confidence:.2f})")

