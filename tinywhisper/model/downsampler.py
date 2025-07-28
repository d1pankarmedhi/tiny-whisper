import torch.nn as nn

class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2):
        super(DownsampleBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=1)
        self.norm = nn.LayerNorm(out_channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        # x (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.gelu(x)
        return x