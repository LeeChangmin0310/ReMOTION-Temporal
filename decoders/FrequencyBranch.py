#############################################
# Frequency Branch: Learnable Continuous Wavelet-based Frequency Features
#############################################

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CWConv(nn.Module):
    """
    A learnable continuous wavelet convolution layer.
    Generates a set of wavelet filters over a specified frequency range.
    """
    def __init__(self, first_freq, last_freq, filter_n, kernel_size, in_channels=1):
        super(CWConv, self).__init__()
        if in_channels != 1:
            raise ValueError("CWConv supports only one input channel")
        self.first_freq = first_freq
        self.last_freq = last_freq
        self.filter_n = filter_n
        self.kernel_size = kernel_size
        self.omega = 5.15
        self.a_ = nn.Parameter(torch.tensor([float(x / 100) for x in range(first_freq, last_freq + 1)]).view(-1, 1))
        self.b_ = torch.tensor(self.omega)

    def forward(self, waveforms):
        """
        Args:
          waveforms: Input tensor of shape (batch, 1, seq_length)
        Returns:
          Output tensor after applying the wavelet filters (batch, filter_n, seq_length)
        """
        device = waveforms.device
        M = self.kernel_size
        x = torch.arange(0, M, device=device, dtype=waveforms.dtype) - (M - 1.0) / 2
        s = (2.5 * self.b_) / (torch.clamp(self.a_, min=1e-7) * 2 * math.pi)
        x = x / s
        wavelet = torch.cos(self.b_ * x) * torch.exp(-0.5 * x ** 2) * (math.pi ** (-0.25))
        output = torch.sqrt(1 / s) * wavelet
        filters = output.view(self.filter_n, 1, M)
        out = F.conv1d(waveforms, filters, stride=1, padding=(M - 1) // 2)
        return out

class FrequencyBranch(nn.Module):
    """
    Extracts spectral features from the recovered rPPG signal using a learnable continuous wavelet convolution.
    For emotion recognition, we focus on the frequency band that captures the heart rate, roughly 1-3 Hz.
    
    Input:
      x: rPPG signal tensor of shape (batch, seq_length)
    Output:
      Embedding tensor of shape (batch, embedding_dim)
    """
    def __init__(self, first_freq=1, last_freq=3, kernel_size=31, embedding_dim=256):
        super(FrequencyBranch, self).__init__()
        n_filter = last_freq - first_freq + 1
        self.cwconv = CWConv(first_freq, last_freq, filter_n=n_filter, kernel_size=kernel_size, in_channels=1)
        self.feature_channels = 64

        self.feature_extractor = nn.Sequential(
            nn.BatchNorm1d(n_filter),
            nn.Conv1d(n_filter, self.feature_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.feature_channels, self.feature_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Sequential(
            nn.Linear(self.feature_channels, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim)
        )

    def forward(self, x):
        """
        Args:
          x: Input rPPG signal tensor of shape (batch, seq_length)
        Returns:
          embedding: Tensor of shape (batch, embedding_dim)
        """
        x = x.unsqueeze(1)  # (batch, 1, seq_length)
        x = self.cwconv(x)  # (batch, n_filter, seq_length)
        x = self.feature_extractor(x)  # (batch, feature_channels, 1)
        x = x.squeeze(-1)  # (batch, feature_channels)
        embedding = self.fc(x)  # (batch, embedding_dim)
        return embedding
