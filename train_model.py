import os

# Trainer: Where the ✨️ happens.
# TrainingArgs: Defines the set of arguments of the Trainer.
from trainer import Trainer, TrainerArgs

# GlowTTSConfig: all model related values for training, validating and testing.
from TTS.tts.configs.glow_tts_config import GlowTTSConfig

# BaseDatasetConfig: defines name, formatter and path of the dataset.
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.glow_tts import GlowTTS
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

# Custom Trainer class with overridden train_step
class CustomTrainer(Trainer):
    def train_step(self, batch, batch_n_steps, cur_step, loader_start_time):
        # Debugging: Print the keys and contents of the batch
        print("Batch keys:", batch.keys())
        for key, value in batch.items():
            print(f"Batch[{key}]:", value)

        # Extract inputs from the batch
        token_id = batch["token_id"]  # Tokenized text input
        token_id_lengths = batch["token_id_lengths"]  # Lengths of tokenized text sequences
        mel_target = batch["mel"]  # Target mel-spectrogram
        mel_lengths = batch["mel_lengths"]  # Lengths of mel-spectrogram sequences

        # Forward pass
        model_outputs = self.model(
            x=token_id,  # Tokenized text input
            x_lengths=token_id_lengths,  # Lengths of tokenized text sequences
            y=mel_target,  # Target mel-spectrogram
            y_lengths=mel_lengths,  # Lengths of mel-spectrogram sequences
        )

        # Unpack model outputs
        z, log_det, y_mean, y_log_scale, attn, o_dur_log, o_attn_dur = model_outputs

        # Compute loss
        loss = self.criterion(
            z=z,
            log_det=log_det,
            y_mean=y_mean,
            y_log_scale=y_log_scale,
            y_lengths=mel_lengths,
            o_dur_log=o_dur_log,
            o_attn_dur=o_attn_dur,
            x_lengths=token_id_lengths,
        )

        # Debugging: Print a warning if the loss is None
        if loss is None:
            print("Warning: Loss is None!")

        # Return the loss and any additional outputs (e.g., predictions)
        return loss, None  # The second value is typically used for additional outputs

# we use the same path as this script as our training folder.
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "tts")
os.makedirs(output_path, exist_ok=True)  # Ensure the output directory exists

# DEFINE DATASET CONFIG
# Set LJSpeech as our target dataset and define its path.
dataset_config = BaseDatasetConfig(
    formatter="ljspeech",  # Use "ljspeech" or a custom formatter
    meta_file_train="metadata.csv",
    path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "LJSpeech-1.1")
)

# INITIALIZE THE TRAINING CONFIGURATION
config = GlowTTSConfig(
    batch_size=1,
    eval_batch_size=1,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=False,
    test_delay_epochs=-1,
    epochs=1,
    text_cleaner="phoneme_cleaners",
    use_phonemes=False,
    phoneme_language="en-us",
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    print_step=25,
    print_eval=False,
    mixed_precision=False,
    output_path=output_path,
    datasets=[dataset_config],
    eval_split_size=0,
)

# INITIALIZE THE AUDIO PROCESSOR
ap = AudioProcessor.init_from_config(config)

# INITIALIZE THE TOKENIZER
tokenizer, config = TTSTokenizer.init_from_config(config)

# LOAD DATA SAMPLES
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=False,  # Disable evaluation split
)

# INITIALIZE THE MODEL
model = GlowTTS(config, ap, tokenizer, speaker_manager=None)

# INITIALIZE THE CUSTOM TRAINER
trainer = CustomTrainer(
    TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
)

# AND... 3,2,1... 🚀
if __name__ == "__main__":
    import torch
    print("GPU available:", torch.cuda.is_available())  # Check if GPU is available
    trainer.fit()