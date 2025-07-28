import torch.nn as nn

from tinywhisper.model.downsampler import DownsampleBlock
from tinywhisper.model.pos_enc import PositionalEncoding


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim,
        emb_dim=256,
        hidden_dim=1024,
        n_layers=4,
        dropout=0.1,
        downsample=True,
    ):
        super().__init__()

        self.down = (
            DownsampleBlock(input_dim, emb_dim)
            if downsample
            else nn.Linear(input_dim, emb_dim)
        )
        self.pos_enc = PositionalEncoding(emb_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=4,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)

    def forward(self, x):  # x (B, T, 80)
        x = self.down(x)  # (B, T, embd_dim)
        x = self.pos_enc(x)
        x = self.encoder(x)
        return x
