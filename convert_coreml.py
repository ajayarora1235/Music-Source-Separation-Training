import torch
import coremltools as ct
import numpy as np
import librosa
import torch.nn as nn
import os
import soundfile as sf

from einops import rearrange, pack, unpack

from utils.settings import get_model_from_config

config_path = "configs/config_bs_roformer_384_8_2_485100.yaml"
start_check_point = "checkpoints/model_bs_roformer_ep_17_sdr_9.6568.ckpt"

# Load original model config
model, config = get_model_from_config("bs_roformer", config_path)

core_ml_version = True

device='cpu'
state_dict = torch.load(start_check_point, map_location=device, weights_only=True)

# Filter out STFT/ISTFT related weights
stft_related_keys = []
#     'stft_window_fn',
#     'stft_kwargs',
#     'multi_stft_resolution_loss_weight',
#     'multi_stft_resolutions_window_sizes',
#     'multi_stft_n_fft',
#     'multi_stft_window_fn',
#     'multi_stft_kwargs'
# ]

def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

filtered_state_dict = {k: v for k, v in state_dict.items() if not any(stft_key in k for stft_key in stft_related_keys)}

# Verify no STFT-related weights remain
for k in filtered_state_dict.keys():
    assert not any(stft_key in k for stft_key in stft_related_keys), f"Found STFT-related key in filtered weights: {k}"

# Load filtered weights
model.load_state_dict(filtered_state_dict)
model.eval()

# Convert model to half precision before moving to device
model = model.half()

# Set up data for inference
example_path = "/Users/ajayarora/Documents/youtube_songs/test_for_model_sep/HYBS_TIP_TOE.m4a"
sample_rate = getattr(config.audio, 'sample_rate', 44100)
print(f"Processing track: {example_path}")
try:
    mix, sr = librosa.load(example_path, sr=sample_rate, mono=False)
except Exception as e:
    print(f'Cannot read track: {format(example_path)}')
    print(f'Error message: {str(e)}')
    exit()

# If mono audio we must adjust it depending on model
if len(mix.shape) == 1:
    mix = np.expand_dims(mix, axis=0)
    if 'num_channels' in config.audio:
        if config.audio['num_channels'] == 2:
            print(f'Convert mono track to stereo...')
            mix = np.concatenate([mix, mix], axis=0)

# Convert input to tensor and ensure it's in half precision
mix = torch.from_numpy(mix).to(device)
chunk_size = config.audio.chunk_size
num_overlap = config.inference.num_overlap

step = chunk_size // num_overlap
border = chunk_size - step
length_init = mix.shape[-1]

# Add padding for generic mode to handle edge artifacts
if length_init > 2 * border and border > 0:
    mix = nn.functional.pad(mix, (border, border), mode="reflect")

# Convert to half precision after padding
mix = mix.half()

# Extract first chunk and apply padding if necessary
part = mix[:, :chunk_size].to(device)
chunk_len = part.shape[-1]
if chunk_len > chunk_size // 2:
    # Convert to float32 for reflection padding
    part = part.float()
    part = nn.functional.pad(part, (0, chunk_size - chunk_len), mode="reflect")
    part = part.half()
else:
    part = nn.functional.pad(part, (0, chunk_size - chunk_len), mode="constant", value=0)

# Pre-compute STFT for input
n_fft = config.model.stft_n_fft
hop_length = config.model.stft_hop_length
win_length = config.model.stft_win_length
window = torch.hann_window(win_length).to(device).half()

# Convert to float32 for STFT
part_float32 = part.float()

raw_audio = part_float32

from einops import rearrange
if raw_audio.ndim == 2:
    raw_audio = rearrange(raw_audio, 'b t -> b 1 t')

print(f"Raw audio shape: {raw_audio.shape}")

channels = raw_audio.shape[0]
assert channels == 2, 'stereo needs to be set to True if passing in audio signal that is stereo (channel dimension of 2). also need to be False if mono (channel dimension of 1)'

# to stft
raw_audio, batch_audio_channel_packed_shape = pack_one(raw_audio, '* t')


# Compute STFT
stft_input = torch.stft(
    part_float32,
    n_fft=n_fft,
    hop_length=hop_length,
    win_length=win_length,
    window=window,
    return_complex=True
)

stft_input = torch.view_as_real(stft_input)
stft_input = stft_input.half().contiguous()  # Ensure STFT output is in half precision

stft_input = unpack_one(stft_input, batch_audio_channel_packed_shape, '* f t c').contiguous()


# Add batch dimension for model input
if core_ml_version:
    stft_input = stft_input
else:
    arr = part.unsqueeze(0).contiguous()

# Export the model with example data
if core_ml_version:
    # exported_program = torch.export.export(model, (stft_input,))
    traced_model = torch.jit.trace(model, stft_input)
else:
    exported_program = torch.export.export(model, (arr,))
    #traced_model = torch.jit.trace(model, arr)

# Test the model before CoreML conversion
# print("Testing model output...")
# with torch.no_grad():
#     # Get model output
#     stft_output = model(stft_input)
    
#     # Reshape output back to separate stems and channels
#     b, n, f, t, _ = stft_output.shape
#     stft_output = stft_output.reshape(b, n // model.audio_channels, model.audio_channels, f, t, 2)
    
#     # Convert to complex and ensure float32
#     stft_output = torch.view_as_complex(stft_output.float())
    
#     # Prepare for ISTFT
#     stft_output = stft_output.squeeze(0)  # Remove batch dimension
    
#     # Create output directory if it doesn't exist
#     os.makedirs("test_outputs", exist_ok=True)
    
#     # Process each stem
#     for stem_idx in range(stft_output.shape[0]):
#         stem_audio = stft_output[stem_idx]
        
#         # ISTFT for each channel
#         for ch_idx in range(stem_audio.shape[0]):
#             # Get the complex STFT for this channel
#             ch_stft = stem_audio[ch_idx]
            
#             # Compute ISTFT
#             audio = torch.istft(
#                 ch_stft,
#                 n_fft=n_fft,
#                 hop_length=hop_length,
#                 win_length=win_length,
#                 window=window.float(),  # Ensure window is float32
#                 return_complex=False,
#                 length=part.shape[-1]
#             )
            
#             # Convert to numpy and save
#             audio_np = audio.cpu().numpy()
#             stem_name = f"stem_{stem_idx}_channel_{ch_idx}"
#             output_path = os.path.join("test_outputs", f"{stem_name}.wav")
#             sf.write(output_path, audio_np, sample_rate)
#             print(f"Saved {output_path}")

print("Model test complete. Proceeding with CoreML conversion...")

model_from_trace = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=stft_input.shape)],
)

print('Model converted to CoreML')

# Convert to Core ML program using the Unified Conversion API.
# model_from_export = ct.convert(
#     exported_program,
#     minimum_deployment_target=ct.target.iOS16,  # Set minimum deployment target to iOS 16
#     compute_units=ct.ComputeUnit.ALL,  # Use all available compute units
#     compute_precision=ct.precision.FLOAT16
# )

# Save the converted model.
model_from_trace.save("bs_roformer_reg.mlpackage")
