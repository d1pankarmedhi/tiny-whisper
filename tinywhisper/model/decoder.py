import torch.nn as nn

from tinywhisper.model.pos_enc import PositionalEncoding


class Decoder(nn.Module):
    def __init__(
        self, vocab_size, embd_dim=256, hidden_dim=1024, num_layers=4, dropout=0.1
    ):
        super(Decoder, self).__init__()
        self.token_embd = nn.Embedding(vocab_size, embd_dim)
        self.pos_enc = PositionalEncoding(embd_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embd_dim,
            nhead=4,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(embd_dim, vocab_size)  # (256, vocabsize)

    def forward(
        self, x, memory, target_mask=None, memory_mask=None
    ):  # memory -> encoder output
        # x: (B, T_text)
        # memory: (B, T_enc, D)
        x = self.token_embd(x)  # (B, T_text, D)
        x = self.pos_enc(x)
        out = self.decoder(x, memory, tgt_mask=target_mask, memory_mask=memory_mask)
        out = self.out_proj(out)  # (B, T_text, vocabsize)
        return out
