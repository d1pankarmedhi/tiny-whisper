import torch.nn as nn

from tinywhisper.config import Config
from tinywhisper.model.decoder import Decoder
from tinywhisper.model.encoder import Encoder


class TinyWhisper(nn.Module):
    def __init__(
        self,
        config: Config,
    ):
        super(TinyWhisper, self).__init__()
        self.encoder = Encoder(
            input_dim=config.audio.n_mels,
            emb_dim=config.model.encoder_dim,
            n_layers=config.model.num_encoder_layers,
        )
        self.decoder = Decoder(
            config.model.vocab_size,
            config.model.decoder_dim,
            num_layers=config.model.num_decoder_layers,
        )

    def forward(self, x, target_tokens, target_mask=None):
        # x: (B, T_audio, 80)
        # target_tokens: (B, T_text)
        memory = self.encoder(x)  # (B, T_enc, D)
        logits = self.decoder(target_tokens, memory, target_mask)  # (B, T_text, vocab)
        return logits
