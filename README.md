<div align="center">
  <h1>TinyWhisper</h1>
  <p> A minimal, efficient encoder-decoder transformer model for speech-to-text (ASR) tasks. Inspired by OpenAI's Whisper, designed for research and educational purposes.</p>

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white) ![Python](https://img.shields.io/badge/Python-blue.svg?style=flat&logo=python&logoColor=white)

</div>

It is a lightweight automatic speech recognition (ASR) system. It follows the encoder-decoder transformer paradigm, processing audio features and generating transcriptions. The project aims to provide a simple, readable codebase for understanding and experimenting with modern ASR techniques.

## Model Architecture

<div align="center">
<img width="400" alt="Image" src="https://github.com/user-attachments/assets/9d29d0ab-cf38-4fd1-8fca-0faaeeb972cc" />
<p>Fig: Encoder-Decoder ASR Model Architecture</p>
</div>

- **Encoder**: Processes input audio features (e.g., log-mel spectrograms) and produces hidden, contextual representations.
- **Decoder**: Autoregressively generates text tokens from the encoder's output.
- **Positional Encoding**: Used in both encoder and decoder to provide sequence order information.
- **Downsampler**: Reduces the temporal resolution of input features for efficiency.

## Tokenizer

The tokenizer is based on Byte Pair Encoding (BPE), similar to Whisper. It converts text to token IDs and vice versa, supporting multilingual and special tokens as needed.

<div align="center">
<img src="https://github.com/user-attachments/assets/dbcad90f-0d78-4407-a48d-4973027fb9b2" width="400" />
<p></p>Fig: Tokenization process</p>
</div>

## Data Preprocessing

### Audio Processing

Audio or Sound is bascially air pressure that varies over time. It is the change in atmospheric presure caused by the vibration of air molecules. These fluctuations create regions of high and low pressure, which we perceive as sound waves. The frequency of these fluctuations determines the pitch of the sound, while the amplitude determines its loudness.

<div align="center">
<img height="200" alt="Image" src="https://github.com/user-attachments/assets/5f0da975-5dc4-46e9-ab1a-6e7899fef32b" />
<p>Fig: Waveform of a sound signal</p>
</div>

For ease of processing, these audio signals are converted into a spectrogram, more precisely a log-mel spectrogram. It captures the frequence-time-intensity representation of the audio signal, making it suitable for input to the model.

<div align="center">
<img height="200" alt="Image" src="https://github.com/user-attachments/assets/ccd6f8ec-a574-4668-92e9-4e03984e793b" />
<p>Fig: Log-Mel Spectrogram of a sound signal</p>
</div>

This helps in filtering out noise and irrelevant sounds from audio sources. It ensures words spoken by different people, man or woman, creates a similar spectrogram, making it easier for the model to learn and generalize.

### Text Processing

The corresonsing audio transcript is tokenized into a sequence of tokens. For tokenization, we use a Byte Pair Encoding (BPE) tokenizer, which is efficient for handling large vocabularies and multilingual text.

For example, the **Start-of-Sequence (SOS)** token is used to indicate the beginning of a transcription, and the **End-of-Sequence (EOS)** token indicates its end. The tokenizer also handles special tokens like padding and unknown words.

```
labels: [50257, 32, 1862, 2576, 12049, 477, 287, 11398, 318, 5055, 319, 257, 13990, 290, 2045, 379, 257, 8223, 50258]
text: <SOS>A young girl dressed all in pink is standing on a fence and looking at a horse<EOS>
```

## Training Process

Training scripts and utilities are provided in the `tinywhisper/train/` directory:

- `train.py`: Main training loop, data loading, and optimization
- Supports custom datasets and data augmentation
- Configurable via `tinywhisper/config/config.py`

### Steps:

1. Prepare your dataset (audio files and transcripts)
2. Configure training parameters in `config.py`
3. Run the training script:
   ```bash
   python -m tinywhisper.train.train
   ```

## Evaluation

Evaluation scripts are in `tinywhisper/eval/`:

- `evaluation.py`: Computes WER/CER and other metrics on test data

## Usage

You can use the model for inference after training:

- Load a trained checkpoint
- Use the inference utilities in `tinywhisper/inference/`

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
