from models.stft_utils import STFT_Process
import torch
import soundfile as sf
import librosa


test_audio = 'HYBS_TIP_TOE.m4a'                     # Specify the path to your audio file.
save_reconstructed_audio = 'HYBS_TIP_TOE_STFT.wav'      # Save the reconstructed.
save_reconstructed_audio_torch = 'HYBS_TIP_TOE_STFT_torch.wav'      # Save the reconstructed.

# Configuration Parameters
MAX_SIGNAL_LENGTH = 1024        # Maximum number of frames for audio length after STFT. Use larger values for long audio inputs (e.g., 4096).
INPUT_AUDIO_LENGTH = 5120       # Length of the audio input signal (in samples) for static axis export. Should be a multiple of NFFT.
WINDOW_TYPE = 'hann'            # Type of window function used in the STFT.  Support: ['bartlett', 'blackman', 'hamming', 'hann', 'kaiser']
NFFT = 512                      # Number of FFT components for the STFT process.
WINDOW_LENGTH = 512             # Match NFFT for best results
WIN_LENGTH = 512                # Length of windowing - match NFFT
HOP_LENGTH = 256                # Use 50% overlap (NFFT/2) for standard processing
SAMPLE_RATE = 48000            # Target sample rate.
STFT_TYPE = "stft_B"            # stft_A: output real_part only;  stft_B: outputs real_part & imag_part
ISTFT_TYPE = "istft_C"          # istft_A: Inputs = [magnitude, phase];  istft_B: Inputs = [magnitude, real_part, imag_part], The dtype of imag_part is float format.

# Load the audio
audio_data, sr = librosa.load(test_audio, sr=SAMPLE_RATE, mono=False)  # mono=False for stereo
print(f"Original audio info - Sample rate: {sr}, Shape: {audio_data.shape}")

# Convert to torch tensor - keep native librosa format
audio = torch.tensor(audio_data, dtype=torch.float32)

print("Audio shape:", audio.shape)
print(f"Audio duration in seconds: {audio.shape[-1] / SAMPLE_RATE:.2f}")

# Create window function for STFT
window = torch.hann_window(WIN_LENGTH) if WINDOW_TYPE == 'hann' else torch.hamming_window(WIN_LENGTH)

def stft_transform(x):
    print("STFT input shape:", x.shape)
    result = torch.stft(
        x,
        n_fft=NFFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window,
        return_complex=True,
        center=True,
        pad_mode='reflect'
    )
    print("STFT output shape:", result.shape)
    return result.real, result.imag

custom_istft = STFT_Process(
    model_type=ISTFT_TYPE, 
    n_fft=NFFT,
    win_length=WIN_LENGTH,
    hop_len=HOP_LENGTH, 
    max_frames=MAX_SIGNAL_LENGTH, 
    window_type=WINDOW_TYPE
).eval()

def istft_transform(x, target_length):
    print("ISTFT input shape:", x.shape)
    result = torch.istft(
        x,
        n_fft=NFFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window,
        center=True,
        length=target_length  # Use the exact target length
    )
    print("ISTFT output shape:", result.shape)
    return result

# Process audio in chunks
print("Initial audio shape:", audio.shape)
total_length = audio.shape[-1]  # Last dimension is always time
reconstructed_chunks = []
reconstructed_chunks_torch = []

for start_idx in range(0, total_length, INPUT_AUDIO_LENGTH):
    end_idx = min(start_idx + INPUT_AUDIO_LENGTH, total_length)
    # Slice along the time dimension (last dimension)
    if audio.dim() == 1:  # Mono
        audio_chunk = audio[start_idx:end_idx]
    else:  # Stereo
        audio_chunk = audio[:, start_idx:end_idx]
    
    print("\nProcessing chunk:")
    print("Chunk shape:", audio_chunk.shape)
    chunk_length = audio_chunk.shape[-1]  # Store the original chunk length
    
    # Process chunk through STFT
    real_part, imag_part = stft_transform(audio_chunk)
    print("Real part shape:", real_part.shape)
    print("Imag part shape:", imag_part.shape)
    magnitude = torch.sqrt(real_part**2 + imag_part**2)
    print("Magnitude shape:", magnitude.shape)
    
    # Reconstruct chunk through ISTFT
    reconstructed_chunk = custom_istft(magnitude, real_part, imag_part)
    print("Custom ISTFT output shape:", reconstructed_chunk.shape)
    reconstructed_chunks.append(reconstructed_chunk)

    ## reconstruct using torch.istft with correct length
    reconstructed_chunk = istft_transform(torch.complex(real_part, imag_part), chunk_length)
    print("Torch ISTFT output shape:", reconstructed_chunk.shape)
    reconstructed_chunks_torch.append(reconstructed_chunk)

# Concatenate all chunks along time dimension
print("\nBefore concatenation:")
print("Number of chunks:", len(reconstructed_chunks))
print("First chunk shape:", reconstructed_chunks[0].shape)
print("First torch chunk shape:", reconstructed_chunks_torch[0].shape)
print("Custom ISTFT sample values:", reconstructed_chunks[0].flatten()[:10])
print("Torch ISTFT sample values:", reconstructed_chunks_torch[0].flatten()[:10])

# Handle shape mismatches for custom ISTFT
processed_chunks = []
for chunk in reconstructed_chunks:
    print(f"Processing chunk shape: {chunk.shape}")
    # If custom ISTFT returns [2, 1, time], squeeze the middle dimension
    if chunk.dim() == 3 and chunk.shape[1] == 1:
        chunk = chunk.squeeze(1)  # [2, 1, time] -> [2, time]
    processed_chunks.append(chunk)

# Concatenate along the time dimension (last dimension)
audio_reconstructed = torch.cat(processed_chunks, dim=-1)
audio_reconstructed_torch = torch.cat(reconstructed_chunks_torch, dim=-1)

print("\nAfter concatenation (before int16 conversion):")
print("audio_reconstructed shape:", audio_reconstructed.shape)
print("audio_reconstructed_torch shape:", audio_reconstructed_torch.shape)
print("audio_reconstructed range:", audio_reconstructed.min().item(), "to", audio_reconstructed.max().item())
print("audio_reconstructed_torch range:", audio_reconstructed_torch.min().item(), "to", audio_reconstructed_torch.max().item())

# Convert to int16 - but first normalize if needed
if audio_reconstructed.abs().max() > 1.0:
    print("Normalizing custom ISTFT output")
    audio_reconstructed = audio_reconstructed / audio_reconstructed.abs().max()

if audio_reconstructed_torch.abs().max() > 1.0:
    print("Normalizing torch ISTFT output") 
    audio_reconstructed_torch = audio_reconstructed_torch / audio_reconstructed_torch.abs().max()

# Convert to int16
audio_reconstructed = (audio_reconstructed * 32767).to(torch.int16)
audio_reconstructed_torch = (audio_reconstructed_torch * 32767).to(torch.int16)

print("\nFinal shapes:")
print("audio_reconstructed shape:", audio_reconstructed.shape)
print("audio_reconstructed_torch shape:", audio_reconstructed_torch.shape)
print("Original audio shape:", audio.shape)

# Ensure we're not adding extra samples
if audio_reconstructed.shape[-1] != audio.shape[-1]:
    print("WARNING: Reconstructed audio length doesn't match input!")
    print(f"Input length: {audio.shape[-1]}, Reconstructed length: {audio_reconstructed.shape[-1]}")

# Save audio - handle both mono and stereo naturally
print("\nSaving files...")
if audio_reconstructed.dim() == 1:  # Mono
    sf.write(save_reconstructed_audio, audio_reconstructed.numpy(), SAMPLE_RATE, format='WAV')
else:  # Stereo - transpose to [samples, channels] format
    print("Saving stereo custom ISTFT with shape:", audio_reconstructed.T.shape)
    sf.write(save_reconstructed_audio, audio_reconstructed.T.numpy(), SAMPLE_RATE, format='WAV')

if audio_reconstructed_torch.dim() == 1:  # Mono
    sf.write(save_reconstructed_audio_torch, audio_reconstructed_torch.numpy(), SAMPLE_RATE, format='WAV')
else:  # Stereo - transpose to [samples, channels] format
    print("Saving stereo torch ISTFT with shape:", audio_reconstructed_torch.T.shape)
    sf.write(save_reconstructed_audio_torch, audio_reconstructed_torch.T.numpy(), SAMPLE_RATE, format='WAV')