import torch
import torchaudio
from datasets import load_dataset

from tinywhisper.config.config import Config
from tinywhisper.data.preprocess import get_mel_spectrogram
from tinywhisper.tokenizer import BPETokenizer


class HFDataset:
    def __init__(self, dataset_name: str, config: Config):
        self.config = config
        self.dataset = load_dataset(dataset_name, split="train")
        self.tokenizer = BPETokenizer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        audio_path = sample["audio_filepath"]
        text = sample["text"]

        waveform, sr = torchaudio.load(audio_path)
        assert sr == self.config.audio.sample_rate, "Sample rate mismatch!"

        # Extract log-Mel spectrogram
        features = get_mel_spectrogram(waveform)  # (T, n_mels)

        # Tokenize
        tokens = self.tokenizer.encode(text)
        tokens = (
            [self.config.training.sos_token_id]
            + tokens
            + [self.config.training.eos_token_id]
        )
        token_tensor = torch.tensor(tokens, dtype=torch.long)

        return {
            "input_features": features,
            "input_length": features.shape[0],
            "labels": token_tensor,
            "label_length": len(token_tensor),
        }
