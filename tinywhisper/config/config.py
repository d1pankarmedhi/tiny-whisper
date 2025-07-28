from dataclasses import dataclass


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    n_mels: int = 80
    win_length: int = 400
    hop_length: int = 160
    n_fft: int = 512
    f_min: int = 0
    f_max: int = 8000


@dataclass
class ModelConfig:
    encoder_dim: int = 256
    decoder_dim: int = 256
    hidden_dim: int = 1024
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.1
    batch_first: bool = True
    vocab_size: int = 50260  # 50257 + SOS + EOS + PAD
    pos_embd_max_length = 5000


@dataclass
class TrainingConfig:
    batch_size: int = 16
    num_epochs: int = 2  # increment this as per your need
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    accumulation_steps: int = 2
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.1
    sos_token_id: int = 50257
    eos_token_id: int = 50258
    pad_token_id: int = 50259
    use_mixed_precision: bool = True


@dataclass
class Dataset:
    hf_dataset_name: str = "DTU54DL/common-native"
    hf_split: str = "train"


@dataclass
class Config:
    audio: AudioConfig = AudioConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    dataset: Dataset = Dataset()
