import numpy as np
import torch
import os
from models.stft_utils import STFT_Process
import soundfile as sf

# Directory containing the .npy files
swift_outputs_dir = 'swift_outputs'

# File names
file_3883 = 'demucs_outputs_var_3883.npy'
file_3785 = 'demucs_outputs_var_3785.npy'
file_3806 = 'demucs_outputs_var_3806.npy'

# Load the .npy files
arr_3883 = np.load(os.path.join(swift_outputs_dir, file_3883))
arr_3785 = np.load(os.path.join(swift_outputs_dir, file_3785))
arr_3806 = np.load(os.path.join(swift_outputs_dir, file_3806))

# Print shapes
print(f"Shape of {file_3883}: {arr_3883.shape}")
print(f"Shape of {file_3785}: {arr_3785.shape}")
print(f"Shape of {file_3806}: {arr_3806.shape}")

# Number of nonzeros in var_3883
num_nonzeros_3883 = np.count_nonzero(arr_3883)
print(f"Number of nonzeros in {file_3883}: {num_nonzeros_3883}")

# Number of nonzeros in var_3785 and var_3806
num_nonzeros_3785 = np.count_nonzero(arr_3785)
num_nonzeros_3806 = np.count_nonzero(arr_3806)
print(f"Number of nonzeros in {file_3785}: {num_nonzeros_3785}")
print(f"Number of nonzeros in {file_3806}: {num_nonzeros_3806}")

# Process var_3785 as three chunks and run STFT_Process (istft_C) on each
arr_3785_torch = torch.from_numpy(arr_3785)

# Assume the first dimension is chunk, so split along axis 0
num_chunks = arr_3785_torch.shape[0]
chunks = [arr_3785_torch[i] for i in range(num_chunks)]

# Initialize STFT_Process for istft_C
# Use n_fft and hop_len based on chunk shape
n_fft = (chunks[0].shape[-2] - 1) * 2
hop_len = n_fft // 4
win_length = n_fft
max_frames = chunks[0].shape[-1]
stft_process = STFT_Process(
    model_type='istft_C',
    n_fft=n_fft,
    win_length=win_length,
    hop_len=hop_len,
    max_frames=max_frames,
    window_type='hann'
)

istft_results = []
for i, chunk in enumerate(chunks):
    chunk = chunk.float()
    # chunk: (8, 4098, 241)
    # For each channel, treat first 4 as real, next 4 as imag (if that's the convention)
    # If not, user may need to clarify, but let's try (real, imag) split
    if chunk.shape[0] == 8:
        real = chunk[:4]  # (4, 4098, 241)
        imag = chunk[4:]  # (4, 4098, 241)
        chunk_stacked = torch.cat([real, imag], dim=1)  # (4, 8196, 241)
        # If you want to keep batch as 8, stack real/imag for each channel
        # chunk_stacked = torch.cat([chunk, chunk], dim=1)  # (8, 8196, 241) if all are real/imag pairs
    else:
        raise ValueError("Unexpected chunk shape for stacking real/imag")
    print(f"Chunk {i} input to STFT_Process shape: {chunk_stacked.shape}")
    istft_result = stft_process(chunk_stacked)
    istft_results.append(istft_result)

# Concatenate results along the chunk axis
istft_concat = torch.cat(istft_results, dim=0)
print(f"istft_concat shape before reshape: {istft_concat.shape}")

# Add istft result to var_3806 (after converting arr_3806 to torch tensor)
arr_3806_torch = torch.from_numpy(arr_3806)

# Reshape to (3, 1, 4, 2, -1)
# istft_concat is (12, 1, N) or (12, N)
if istft_concat.ndim == 3:
    istft_concat = istft_concat.squeeze(1)
# Now istft_concat is (12, N)
istft_reshaped = istft_concat.view(3, 1, 4, 2, -1)
print(f"istft_reshaped shape: {istft_reshaped.shape}")

# Trim the last dimension to match arr_3806_torch
target_length = arr_3806_torch.shape[-1]
if istft_reshaped.shape[-1] > target_length:
    istft_reshaped = istft_reshaped[..., :target_length]
elif istft_reshaped.shape[-1] < target_length:
    # Pad if needed
    pad_size = target_length - istft_reshaped.shape[-1]
    istft_reshaped = torch.nn.functional.pad(istft_reshaped, (0, pad_size))

print(f"istft_reshaped shape after trim/pad: {istft_reshaped.shape}")

# Multiply istft_reshaped by 64
istft_reshaped = istft_reshaped * 64

# Ensure shapes match for addition
if istft_reshaped.shape == arr_3806_torch.shape:
    added_result = istft_reshaped + arr_3806_torch
    print("Added istft(var_3785) to var_3806 successfully.")

    # Concatenate the three chunks along the last axis
    # added_result shape: (3, 1, 4, 2, 242550)
    added_concat = added_result.permute(1, 2, 3, 0, 4).reshape(1, 4, 2, -1)  # (1, 4, 2, 242550*3)
    print(f"added_concat shape: {added_concat.shape}")

    os.makedirs('coreml_tests', exist_ok=True)
    for ch in range(4):
        audio = added_concat[0, ch].cpu().numpy()  # shape (2, 242550*3)
        # Normalize to prevent clipping
        max_val = max(abs(audio.max()), abs(audio.min()), 1e-8)
        audio = audio / max_val * 0.99
        sf.write(f'coreml_tests/wav_{ch+1}.wav', audio.T, 44100)
        print(f"Saved coreml_tests/wav_{ch+1}.wav, shape: {audio.shape}")
else:
    print(f"Shape mismatch: istft result shape {istft_reshaped.shape}, var_3806 shape {arr_3806_torch.shape}") 