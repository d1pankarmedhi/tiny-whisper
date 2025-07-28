<div align="center">
  <h1>TinyWhisper</h1>
  <p> A minimal, efficient encoder-decoder transformer model for speech-to-text (ASR) tasks. Inspired by OpenAI's Whisper, designed for research and educational purposes.</p>

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white) ![Python](https://img.shields.io/badge/Python-blue.svg?style=flat&logo=python&logoColor=white)

</div>

`tiny-whisper` is a lightweight automatic speech recognition (ASR) system. It follows the encoder-decoder transformer paradigm, processing audio features and generating transcriptions. The project aims to provide a simple, readable codebase for understanding and experimenting with modern ASR techniques.

## Model Architecture

<div align="center">
<img width="400" alt="Image" src="https://github.com/user-attachments/assets/9d29d0ab-cf38-4fd1-8fca-0faaeeb972cc" />
</div>

- **Encoder**: Processes input audio features (e.g., log-mel spectrograms) and produces hidden, contextual representations.
- **Decoder**: Autoregressively generates text tokens from the encoder's output.
- **Positional Encoding**: Used in both encoder and decoder to provide sequence order information.
- **Downsampler**: Reduces the temporal resolution of input features for efficiency.

The model is implemented in the `tinywhisper/model/` directory:

- `encoder.py`: Encoder transformer implementation
- `decoder.py`: Decoder transformer implementation
- `downsampler.py`: Feature downsampling
- `pos_enc.py`: Positional encoding
- `model.py`: Model wrapper and integration

## Tokenizer

The tokenizer is based on Byte Pair Encoding (BPE), similar to Whisper. It converts text to token IDs and vice versa, supporting multilingual and special tokens as needed.

<div align="center">
<img src="https://github.com/user-attachments/assets/dbcad90f-0d78-4407-a48d-4973027fb9b2" width="400" />
</div>

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
