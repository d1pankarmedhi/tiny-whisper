import torchaudio


def get_mel_spectrogram(
    waveform,
    sr: int = 16000,
    n_ftt: int = 400,
    hop_length: int = 160,
    n_mels: int = 80,
):
    mel_extractor = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_ftt,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    db_transform = torchaudio.transforms.AmplitudeToDB()

    # log-mel spectrogram
    mel = mel_extractor(waveform)
    mel_db = db_transform(mel).squeeze(0).transpose(0, 1)  # (T, n_mels)

    return mel_db
