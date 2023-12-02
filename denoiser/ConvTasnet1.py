import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa

class ConvTasNet(nn.Module):
    def __init__(self, N, L, B, H, P, X, R, frame_length, frame_step):
        super(ConvTasNet, self).__init__()
        
        self.frame_length = frame_length
        self.frame_step = frame_step
        
        # Encoder
        self.encoder = nn.Conv1d(1, N, kernel_size=L, stride=L//2)
        
        # Separation module
        self.separator = TemporalConvNet(N, B, H, P, X, R)
        
        # Decoder
        self.decoder = nn.ConvTranspose1d(N, 1, kernel_size=L, stride=L//2)

    def forward(self, x):
        frames = librosa.effects.frame(x, frame_length=self.frame_length, hop_length=self.frame_step)
        frames = torch.from_numpy(frames).float()

        encoded_frames = self.encoder(frames)
        separated_frames = self.separator(encoded_frames)
        reconstructed_frames = self.decoder(separated_frames)
        
        return reconstructed_frames

class TemporalConvNet(nn.Module):
    def __init__(self, N, B, H, P, X, R):
        super(TemporalConvNet, self).__init__()
        layers = []
        for _ in range(B):
            dilation = 2 ** _ if _ < R else 2 ** ((_ - R) % X)
            layers += [TemporalBlock(N, H, P, kernel_size=3, dilation=dilation)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=dilation, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.layer_norm(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.layer_norm(self.conv2(x)))
        x += residual
        return x