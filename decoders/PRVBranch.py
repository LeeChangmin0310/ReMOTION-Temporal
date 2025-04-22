#############################################
# PRV Branch: Differentiable PRV Branch for End-to-End Training
#############################################

import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SlidingWindowPeakDetector(nn.Module):
    """
    Uses softmax to compute a differentiable approximation of the peak position
    within a sliding window over the input signal.
    """
    def __init__(self, window_size=30, stride=15, temperature=0.15):
        super(SlidingWindowPeakDetector, self).__init__()
        self.window_size = window_size
        self.stride = stride
        self.temperature = temperature

    def forward(self, x):
        """
        Args:
          x: Input tensor of shape (batch, seq_length)
        Returns:
          peaks: Tensor of shape (batch, num_windows) containing predicted peak positions
        """
        batch_size, seq_length = x.size()
        ws = self.window_size
        st = self.stride
        num_windows = (seq_length - ws) // st + 1
        peaks = []
        for i in range(num_windows):
            start = i * st
            end = start + ws
            window = x[:, start:end]  # (batch, window_size)
            softmax_weights = F.softmax(window / self.temperature, dim=1)  # (batch, window_size)
            indices = torch.arange(ws, device=x.device, dtype=x.dtype).unsqueeze(0)  # (1, window_size)
            expected_index = torch.sum(softmax_weights * indices, dim=1)  # (batch,)
            absolute_index = expected_index + start  # Offset to absolute position
            peaks.append(absolute_index.unsqueeze(1))  # (batch, 1)
        peaks = torch.cat(peaks, dim=1)  # (batch, num_windows)
        return peaks

class PRVBranch(nn.Module):
    """
    Approximates HRV features in a differentiable manner.
    It computes time-domain features (Mean RR, RMSSD, SDNN) and frequency-domain
    features (LF and HF power) from the rPPG signal using sliding window peak detection.
    The 5 extracted HRV features are then mapped to an embedding vector via a fully-connected network.

    Input:
      x: rPPG signal tensor of shape (batch, seq_length)
    Output:
      Embedding tensor of shape (batch, embedding_dim)
    """
    def __init__(self, embedding_dim=128, window_size=30, stride=15, temperature=0.1, fs=30.0):
        super(PRVBranch, self).__init__()
        self.fs = fs  # Sampling frequency in Hz
        self.peak_detector = SlidingWindowPeakDetector(window_size, stride, temperature)
        self.embedding_dim = embedding_dim
        # The input dimension is now 5 (mean_rr, rmssd, sdnn, LF, HF)
        self.fc = nn.Sequential(
            nn.Linear(5, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

    def forward(self, x):
        # x: (batch, seq_length)
        peaks = self.peak_detector(x)  # (batch, num_windows)
        # Calculate RR intervals in terms of frame indices
        rr_intervals = peaks[:, 1:] - peaks[:, :-1]  # (batch, num_windows - 1)
        # Convert RR intervals to seconds using the sampling rate
        rr_intervals_sec = rr_intervals / self.fs

        # Time-domain HRV features
        mean_rr = rr_intervals_sec.mean(dim=1, keepdim=True)  # (batch, 1)
        if rr_intervals_sec.size(1) >= 2:
            diff_rr = rr_intervals_sec[:, 1:] - rr_intervals_sec[:, :-1]
            rmssd = torch.sqrt((diff_rr ** 2).mean(dim=1, keepdim=True) + 1e-6)
        else:
            rmssd = torch.zeros_like(mean_rr)
        sdnn = rr_intervals_sec.std(dim=1, keepdim=True)  # (batch, 1)

        # Frequency-domain HRV features using FFT
        n = rr_intervals_sec.size(1)
        # Zero-pad to next power of 2 for better FFT resolution
        n_fft = 2 ** int(np.ceil(np.log2(n)))
        # Compute FFT along time dimension (n)
        fft_rr = torch.fft.rfft(rr_intervals_sec, n=n_fft, dim=1)  # shape: (batch, n_fft//2+1)
        power = torch.abs(fft_rr) ** 2  # Power spectrum

        # Frequency bins corresponding to FFT output
        freqs = torch.fft.rfftfreq(n_fft, d=1/self.fs).to(x.device)  # shape: (n_fft//2+1,)
        # Define LF and HF bands (typical HRV frequency ranges in Hz)
        lf_mask = (freqs >= 0.04) & (freqs < 0.15)
        hf_mask = (freqs >= 0.15) & (freqs <= 0.4)
        # Sum power in LF and HF bands
        lf_power = power[:, lf_mask].sum(dim=1, keepdim=True)
        hf_power = power[:, hf_mask].sum(dim=1, keepdim=True)

        # Concatenate all HRV features: (mean_rr, rmssd, sdnn, lf_power, hf_power)
        hrv_features = torch.cat([mean_rr, rmssd, sdnn, lf_power, hf_power], dim=1)  # (batch, 5)
        embedding = self.fc(hrv_features)  # (batch, embedding_dim)
        return embedding

'''
class SlidingWindowPeakDetector(nn.Module):
    """
    Uses softmax to compute a differentiable approximation of the peak position
    within a sliding window over the input signal.
    """
    def __init__(self, window_size=30, stride=15, temperature=0.15):
        super(SlidingWindowPeakDetector, self).__init__()
        self.window_size = window_size
        self.stride = stride
        self.temperature = temperature

    def forward(self, x):
        """
        Args:
          x: Input tensor of shape (batch, seq_length)
        Returns:
          peaks: Tensor of shape (batch, num_windows) containing predicted peak positions
        """
        batch_size, seq_length = x.size()
        ws = self.window_size
        st = self.stride
        num_windows = (seq_length - ws) // st + 1
        peaks = []
        for i in range(num_windows):
            start = i * st
            end = start + ws
            window = x[:, start:end]  # (batch, window_size)
            softmax_weights = F.softmax(window / self.temperature, dim=1)  # (batch, window_size)
            indices = torch.arange(ws, device=x.device, dtype=x.dtype).unsqueeze(0)  # (1, window_size)
            expected_index = torch.sum(softmax_weights * indices, dim=1)  # (batch,)
            absolute_index = expected_index + start  # Offset to absolute position
            peaks.append(absolute_index.unsqueeze(1))  # (batch, 1)
        peaks = torch.cat(peaks, dim=1)  # (batch, num_windows)
        return peaks

class PRVBranch(nn.Module):
    """
    Approximates HRV features (Mean RR, RMSSD, SDNN) in a differentiable manner using
    sliding window soft-peak detection. The extracted HRV features are then mapped to an
    embedding vector via a fully-connected network.

    Input:
      x: rPPG signal tensor of shape (batch, seq_length)
    Output:
      Embedding tensor of shape (batch, embedding_dim)
    """
    def __init__(self, embedding_dim=128, window_size=30, stride=15, temperature=0.1):
        super(PRVBranch, self).__init__()
        self.peak_detector = SlidingWindowPeakDetector(window_size, stride, temperature)
        self.embedding_dim = embedding_dim
        self.fc = nn.Sequential(
            nn.Linear(3, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

    def forward(self, x):
        # x: (batch, seq_length)
        peaks = self.peak_detector(x)  # (batch, num_windows)
        rr_intervals = peaks[:, 1:] - peaks[:, :-1]  # (batch, num_windows - 1)

        mean_rr = rr_intervals.mean(dim=1, keepdim=True)  # (batch, 1)
        if rr_intervals.size(1) >= 2:
            diff_rr = rr_intervals[:, 1:] - rr_intervals[:, :-1]
            rmssd = torch.sqrt((diff_rr ** 2).mean(dim=1, keepdim=True) + 1e-6)
        else:
            rmssd = torch.zeros_like(mean_rr)
        sdnn = rr_intervals.std(dim=1, keepdim=True)  # (batch, 1)

        prv_features = torch.cat([mean_rr, rmssd, sdnn], dim=1)  # (batch, 3)
        embedding = self.fc(prv_features)  # (batch, embedding_dim)
        return embedding
'''