import numpy as np
import torch

# To export your own STFT process ONNX model, set the following values.
# Next, click the IDE Run button or Launch the cmd to run 'python STFT_Process.py'

DYNAMIC_AXES = True         # Default dynamic axes is input audio (signal) length.
NFFT = 512                  # Number of FFT components for the STFT process
WIN_LENGTH = 400            # Length of the window function (can be different from NFFT)
HOP_LENGTH = 160            # Number of samples between successive frames in the STFT
INPUT_AUDIO_LENGTH = 16000  # Set for static axes. Length of the audio input signal in samples.
MAX_SIGNAL_LENGTH = 2048    # Maximum number of frames for the audio length after STFT processed. Set a appropriate larger value for long audio input, such as 4096.
WINDOW_TYPE = 'hann'      # Type of window function used in the STFT
PAD_MODE = 'reflect'        # Select reflect or constant
STFT_TYPE = "stft_A"        # stft_A: output real_part only;  stft_B: outputs real_part & imag_part
ISTFT_TYPE = "istft_A"      # istft_A: Inputs = [magnitude, phase];  istft_B: Inputs = [magnitude, real_part, imag_part], The dtype of imag_part is float format.
export_path_stft = f"{STFT_TYPE}.onnx"      # The exported stft onnx model save path.
export_path_istft = f"{ISTFT_TYPE}.onnx"    # The exported istft onnx model save path.

# # Precompute constants to avoid calculations at runtime
# HALF_NFFT = NFFT // 2
# STFT_SIGNAL_LENGTH = INPUT_AUDIO_LENGTH // HOP_LENGTH + 1

# # Sanity checks for parameters
# NFFT = min(NFFT, INPUT_AUDIO_LENGTH)
# WIN_LENGTH = min(WIN_LENGTH, NFFT)  # Window length cannot exceed NFFT
# HOP_LENGTH = min(HOP_LENGTH, INPUT_AUDIO_LENGTH)

# Create window function lookup once
WINDOW_FUNCTIONS = {
    'bartlett': torch.bartlett_window,
    'blackman': torch.blackman_window,
    'hamming': torch.hamming_window,
    'hann': torch.hann_window,
    'kaiser': lambda x: torch.kaiser_window(x, periodic=True, beta=12.0)
}
# Define default window function
DEFAULT_WINDOW_FN = torch.hann_window


class STFT_Process(torch.nn.Module):
    def __init__(self, model_type, n_fft=NFFT, win_length=WIN_LENGTH, hop_len=HOP_LENGTH, max_frames=MAX_SIGNAL_LENGTH,
                 window_type=WINDOW_TYPE):
        super(STFT_Process, self).__init__()
        self.model_type = model_type
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_len = hop_len
        self.max_frames = max_frames
        self.window_type = window_type
        self.half_n_fft = n_fft // 2  # Precompute once

        # Create the properly sized window
        window = self.create_padded_window(win_length, n_fft, window_type)

        # Register common buffers for all model types
        self.register_buffer('padding_zero', torch.zeros((1, 1, self.half_n_fft), dtype=torch.float32), persistent=False)

        # Pre-compute model-specific buffers
        if self.model_type in ['stft_A', 'stft_B']:
            # STFT forward pass preparation
            time_steps = torch.arange(n_fft, dtype=torch.float32).unsqueeze(0)
            frequencies = torch.arange(self.half_n_fft + 1, dtype=torch.float32).unsqueeze(1)

            # Calculate omega matrix once
            omega = 2 * torch.pi * frequencies * time_steps / n_fft

            # Register conv kernels as buffers
            self.register_buffer('cos_kernel', (torch.cos(omega) * window.unsqueeze(0)).unsqueeze(1), persistent=False)
            self.register_buffer('sin_kernel', (-torch.sin(omega) * window.unsqueeze(0)).unsqueeze(1), persistent=False)

        if self.model_type in ['istft_A', 'istft_B', 'istft_C']:
            # ISTFT forward pass preparation
            # Pre-compute fourier basis
            fourier_basis = torch.fft.fft(torch.eye(n_fft, dtype=torch.float32))
            fourier_basis = torch.vstack([
                torch.real(fourier_basis[:self.half_n_fft + 1, :]),
                torch.imag(fourier_basis[:self.half_n_fft + 1, :])
            ]).float()

            # Create forward and inverse basis
            forward_basis = window * fourier_basis.unsqueeze(1)
            inverse_basis = window * torch.linalg.pinv((fourier_basis * n_fft) / hop_len).T.unsqueeze(1)

            # Calculate window sum for overlap-add
            n = n_fft + hop_len * (max_frames - 1)
            window_sum = torch.zeros(n, dtype=torch.float32)

            # For window sum calculation, we need the original window (not padded to n_fft)
            original_window = WINDOW_FUNCTIONS.get(window_type, DEFAULT_WINDOW_FN)(win_length).float()
            window_normalized = original_window / original_window.abs().max()

            # Pad the original window to n_fft for overlap-add calculation
            if win_length < n_fft:
                pad_left = (n_fft - win_length) // 2
                pad_right = n_fft - win_length - pad_left
                win_sq = torch.nn.functional.pad(window_normalized ** 2, (pad_left, pad_right), mode='constant',
                                                 value=0)
            else:
                win_sq = window_normalized ** 2

            # Calculate overlap-add weights
            for i in range(max_frames):
                sample = i * hop_len
                window_sum[sample: min(n, sample + n_fft)] += win_sq[: max(0, min(n_fft, n - sample))]

            # Register buffers (non-persistent so they don't get saved in checkpoints)
            self.register_buffer("forward_basis", forward_basis, persistent=False)
            self.register_buffer("inverse_basis", inverse_basis, persistent=False)
            self.register_buffer("window_sum_inv",
                                 n_fft / (window_sum * hop_len + 1e-7), persistent=False)  # Add epsilon to avoid division by zero

    def create_padded_window(self,win_length, n_fft, window_type):
        """Create a window of win_length and pad/truncate to n_fft size"""
        window_fn = WINDOW_FUNCTIONS.get(window_type, DEFAULT_WINDOW_FN)
        window = window_fn(win_length).float()

        if win_length == n_fft:
            return window
        elif win_length < n_fft:
            # Pad the window to n_fft size
            pad_left = (n_fft - win_length) // 2
            pad_right = n_fft - win_length - pad_left
            return torch.nn.functional.pad(window, (pad_left, pad_right), mode='constant', value=0)
        else:
            # Truncate the window to n_fft size (this shouldn't happen with our sanity check above)
            start = (win_length - n_fft) // 2
            return window[start:start + n_fft]
    
    def forward(self, *args, **kwargs):
        # Use direct method calls instead of if-else cascade for better ONNX export
        if self.model_type == 'stft_A':
            return self.stft_A_forward(*args, **kwargs)
        if self.model_type == 'stft_B':
            return self.stft_B_forward(*args, **kwargs)
        if self.model_type == 'istft_A':
            return self.istft_A_forward(*args, **kwargs)
        if self.model_type == 'istft_B':
            return self.istft_B_forward(*args, **kwargs)
        if self.model_type == 'istft_C':
            return self.istft_C_forward(*args, **kwargs)
        # In case none match, raise an error
        raise ValueError(f"Unknown model type: {self.model_type}")

    def stft_A_forward(self, x, pad_mode='reflect' if PAD_MODE == 'reflect' else 'constant'):
        if pad_mode == 'reflect':
            x_padded = torch.nn.functional.pad(x, (self.half_n_fft, self.half_n_fft), mode=pad_mode)
        else:
            x_padded = torch.cat((self.padding_zero, x, self.padding_zero), dim=-1)

        # Single conv operation
        return torch.nn.functional.conv1d(x_padded, self.cos_kernel, stride=self.hop_len)

    def stft_B_forward(self, x, pad_mode='reflect' if PAD_MODE == 'reflect' else 'constant'):
        if pad_mode == 'reflect':
            x_padded = torch.nn.functional.pad(x, (self.half_n_fft, self.half_n_fft), mode=pad_mode)
        else:
            x_padded = torch.cat((self.padding_zero, x, self.padding_zero), dim=-1)

        # Perform convolutions
        real_part = torch.nn.functional.conv1d(x_padded, self.cos_kernel, stride=self.hop_len)
        image_part = torch.nn.functional.conv1d(x_padded, self.sin_kernel, stride=self.hop_len)

        return real_part, image_part

    def istft_A_forward(self, magnitude, phase):
        # Pre-compute trig values
        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)

        # Prepare input for transposed convolution
        complex_input = torch.cat((magnitude * cos_phase, magnitude * sin_phase), dim=1)

        # Perform transposed convolution
        inverse_transform = torch.nn.functional.conv_transpose1d(
            complex_input,
            self.inverse_basis,
            stride=self.hop_len,
            padding=0,
        )

        # Apply window correction
        output_len = inverse_transform.size(-1)
        start_idx = self.half_n_fft
        end_idx = output_len - self.half_n_fft

        return inverse_transform[:, :, start_idx:end_idx] * self.window_sum_inv[start_idx:end_idx]

    def istft_B_forward(self, magnitude, real, imag):
        # Calculate phase using atan2
        phase = torch.atan2(imag, real)

        # Pre-compute trig values directly instead of calling istft_A_forward
        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)

        # Prepare input for transposed convolution
        complex_input = torch.cat((magnitude * cos_phase, magnitude * sin_phase), dim=1)

        # Perform transposed convolution
        inverse_transform = torch.nn.functional.conv_transpose1d(
            complex_input,
            self.inverse_basis,
            stride=self.hop_len,
            padding=0,
        )

        # Apply window correction
        output_len = inverse_transform.size(-1)
        start_idx = self.half_n_fft
        end_idx = output_len - self.half_n_fft

        return inverse_transform[:, :, start_idx:end_idx] * self.window_sum_inv[start_idx:end_idx]

    def istft_C_forward(self, z, length=None, hop_length=None, n_fft=None):
        # z should be a real tensor with shape [batch, 2*freqs, frames]
        # where the first half are real parts and second half are imaginary parts
        
        # Use provided parameters or fall back to instance defaults
        hop_length = hop_length if hop_length is not None else self.hop_len
        n_fft = n_fft if n_fft is not None else self.n_fft
        
        # Check if we can use precomputed buffers
        print(hop_length, self.hop_len, n_fft, self.n_fft, "HOP_LENGTH, N_FFT")
        # Use precomputed buffers for efficiency, but ensure they're on the right device
        device = z.device

        # Split the input tensor into real and imaginary parts
        # real_part = z[:, :z.size(1)//2, :]
        # imag_part = z[:, z.size(1)//2:, :]
        # z = torch.complex(real_part, imag_part)

        #result = torch.istft(z.cpu(), n_fft=n_fft, hop_length=hop_length, length=length)
        #return result
        
        # Move buffers to the same device as input if needed
        # if self.inverse_basis.device != device:
        #     self.inverse_basis = self.inverse_basis.to(device)
        #     self.window_sum_inv = self.window_sum_inv.to(device)
        
        # inverse_basis = self.inverse_basis
        # window_sum_inv = self.window_sum_inv
        
        # Perform transposed convolution
        inverse_transform = torch.nn.functional.conv_transpose1d(
            z.cpu(),
            self.inverse_basis.cpu(),
            stride=hop_length,
            padding=0,
        )

        # Apply window correction and trimming
        inverse_transform = inverse_transform.to(device)
        output_len = inverse_transform.size(-1)
        half_n_fft = n_fft // 2
        start_idx = half_n_fft
        
        if length is not None:
            # Use provided length
            end_idx = min(start_idx + length, output_len)
        else:
            # Use default trimming
            end_idx = output_len - half_n_fft
        
        # Ensure we don't go beyond the available length
        end_idx = min(end_idx, output_len)
        if start_idx >= output_len or start_idx >= end_idx:
            # Handle edge case where output is too short
            return inverse_transform[:, :, :max(1, output_len)]
        
        corrected_output = inverse_transform[:, :, start_idx:end_idx]
        
        # Apply window sum correction only to the valid range
        window_correction = self.window_sum_inv[start_idx:end_idx]
        if corrected_output.size(-1) != window_correction.size(0):
            # Adjust window correction to match output size
            min_len = min(corrected_output.size(-1), window_correction.size(0))
            corrected_output = corrected_output[:, :, :min_len]
            window_correction = window_correction[:min_len]
        
        # Ensure both tensors are on the same device before multiplication
        window_correction = window_correction.to(corrected_output.device)
        return corrected_output * window_correction