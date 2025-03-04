import os

from trainer import Trainer, TrainerArgs

from TTS.utils.audio import AudioProcessor
from TTS.vocoder.configs import HifiganConfig
from TTS.vocoder.datasets.preprocess import load_wav_data
from TTS.vocoder.models.gan import GAN

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "vocoder")

config = HifiganConfig(
    batch_size=1,
    eval_batch_size=1,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=5,
    epochs=2,
    seq_len=8192,
    pad_short=2000,
    use_noise_augment=True,
    eval_split_size=1,
    print_step=25,
    print_eval=False,
    mixed_precision=False,
    lr_gen=1e-4,
    lr_disc=1e-4,
    data_path=os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "LJSpeech-1.1/wavs")),
    output_path=output_path,
)

# init audio processor
ap = AudioProcessor(**config.audio.to_dict())

# load training samples
eval_samples, train_samples = load_wav_data(config.data_path, config.eval_split_size)

# init model
model = GAN(config, ap)

# init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
)

if __name__ == "__main__":
    trainer.fit()