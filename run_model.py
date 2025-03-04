import torch
from TTS.api import TTS
from TTS.utils.audio import AudioProcessor
import scipy.io.wavfile as wav
import json

# Paths to your trained TTS model and configuration file
model_path = "models/tts/run-March-04-2025_04+46PM-8da737e/best_model_8.pth"
config_path = "models/tts/run-March-04-2025_04+46PM-8da737e/config.json"

# Load your trained TTS model
tts = TTS(model_path=model_path, config_path=config_path, progress_bar=False)

# Paths for the vocoder model and its config
vocoder_model_path = "models/vocoder/run-March-04-2025_05+38PM-8da737e/best_model_9.pth"
vocoder_config_path = "models/vocoder/run-March-04-2025_05+38PM-8da737e/config.json"

# Load the vocoder's configuration
with open(vocoder_config_path, 'r') as f:
    vocoder_config = json.load(f)

# Load the vocoder model (use the correct loading method for your vocoder)
vocoder = torch.load(vocoder_model_path)

# Generate speech from text using the TTS object
text = "This is voice cloning."

# Explicitly pass the speaker and language arguments
# If there is only one speaker, pass `None` for the speaker and language
file_path = 'output_cloned_voice.wav'
tts.tts_to_file(text=text, 
                speaker=None, 
                language=None, 
                file_path=file_path, 
                speed=1.0,
                split_sentences=True)