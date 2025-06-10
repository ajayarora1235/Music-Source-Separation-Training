import torch
import coremltools as ct
import numpy as np
import librosa
import torch.nn as nn
import os
import soundfile as sf
import argparse
from typing import Tuple, Dict, Any
from einops import rearrange, pack, unpack
from demucs.htdemucs import HTDemucs
from fractions import Fraction
#from executorch.backends.apple.coreml.partition import CoreMLPartitioner
#from executorch.exir import to_edge_transform_and_lower

from utils.settings import get_model_from_config

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Convert model to CoreML format')
    parser.add_argument('--config_path', type=str, required=True,
                      help='Path to the model config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to the model checkpoint')
    parser.add_argument('--model_type', type=str, required=True,
                      help='Type of model (e.g., bs_roformer)')
    parser.add_argument('--use_stft', action='store_true',
                      help='Whether to use STFT preprocessing')
    parser.add_argument('--example_audio', type=str, required=True,
                      help='Path to example audio file for conversion')
    parser.add_argument('--output_path', type=str, default='model.mlpackage',
                      help='Path to save the CoreML model')
    parser.add_argument('--half', action='store_true',
                      help='Whether to convert model to half precision')
    return parser.parse_args()

def load_model_and_config(model_type: str, config_path: str, checkpoint_path: str, 
                         use_stft: bool = True, half: bool = False) -> Tuple[torch.nn.Module, Any]:
    """Load model and config, optionally filtering STFT weights."""
    model, config = get_model_from_config(model_type, config_path)
    
    device = 'cpu'
    
    if model_type in ['htdemucs', 'apollo']:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
        # Fix for htdemucs pretrained models
        if 'state' in state_dict:
            state_dict = state_dict['state']
        # Fix for apollo pretrained models
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
    else:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    
    if use_stft:
        # Filter out STFT/ISTFT related weights
        stft_related_keys = [
            'stft_window_fn',
            'stft_kwargs',
            'multi_stft_resolution_loss_weight',
            'multi_stft_resolutions_window_sizes',
            'multi_stft_n_fft',
            'multi_stft_window_fn',
            'multi_stft_kwargs'
        ]
        filtered_state_dict = {k: v for k, v in state_dict.items() 
                             if not any(stft_key in k for stft_key in stft_related_keys)}
        
        print('found number of keys: ', len(filtered_state_dict.keys()))
        
        # Verify no STFT-related weights remain
        for k in filtered_state_dict.keys():
            assert not any(stft_key in k for stft_key in stft_related_keys), \
                f"Found STFT-related key in filtered weights: {k}"
    else:
        filtered_state_dict = state_dict
    
    model.load_state_dict(filtered_state_dict)
    model.eval()
    return model.half() if half else model, config

def prepare_audio(audio_path: str, config: Any, half: bool = False) -> torch.Tensor:
    """Load and prepare audio for model input."""
    sample_rate = getattr(config.audio, 'sample_rate', 44100)
    print(f"Processing track: {audio_path}")
    
    try:
        mix, sr = librosa.load(audio_path, sr=sample_rate, mono=False)
    except Exception as e:
        print(f'Cannot read track: {audio_path}')
        print(f'Error message: {str(e)}')
        exit()

    # Handle mono audio
    if len(mix.shape) == 1:
        mix = np.expand_dims(mix, axis=0)
        if 'num_channels' in config.audio:
            if config.audio['num_channels'] == 2:
                print(f'Convert mono track to stereo...')
                mix = np.concatenate([mix, mix], axis=0)

    return torch.from_numpy(mix).half() if half else torch.from_numpy(mix)

def process_audio_chunk(audio: torch.Tensor, config: Any, half: bool = False) -> torch.Tensor:
    """Process audio chunk with appropriate padding and STFT if needed."""
    chunk_size = config.audio.chunk_size
    num_overlap = config.inference.num_overlap
    step = chunk_size // num_overlap
    border = chunk_size - step
    
    # Convert to float32 for padding operations
    original_dtype = audio.dtype
    audio = audio.float()
    
    # Add padding for generic mode
    if audio.shape[-1] > 2 * border and border > 0:
        audio = nn.functional.pad(audio, (border, border), mode="reflect")
    
    # Extract first chunk
    part = audio[:, :chunk_size]
    chunk_len = part.shape[-1]
    
    if chunk_len > chunk_size // 2:
        part = nn.functional.pad(part, (0, chunk_size - chunk_len), mode="reflect")
    else:
        part = nn.functional.pad(part, (0, chunk_size - chunk_len), mode="constant", value=0)
    
    # Convert back to original dtype
    if half:
        part = part.half()
    else:
        part = part.to(original_dtype)
    
    return part

def compute_stft(audio: torch.Tensor, config: Any, half: bool = False) -> torch.Tensor:
    """Compute STFT of the audio input."""
    n_fft = config.model.stft_n_fft
    hop_length = config.model.stft_hop_length
    win_length = config.model.stft_win_length
    window = torch.hann_window(win_length).half() if half else torch.hann_window(win_length)
    
    # Ensure audio is in float32 for STFT
    audio_float32 = audio.float()
    
    # Compute STFT
    stft_input = torch.stft(
        audio_float32,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True
    )
    
    stft_input = torch.view_as_real(stft_input)
    return stft_input.half() if half else stft_input.contiguous()

def convert_to_coreml(model: torch.nn.Module, example_input: torch.Tensor, 
                     output_path: str) -> None:
    """Convert PyTorch model to CoreML format."""
    model.eval()  # Ensure model is in eval mode
    
    # Use script instead of trace for better handling of control flow
    example_inputs = (example_input,)
    exported_program = torch.export.export(
        model, 
        example_inputs,
    )

    # executorch_program = to_edge_transform_and_lower(
    #     exported_program,
    #     partitioner = [CoreMLPartitioner()]
    # ).to_executorch()

    #exported_program = torch.jit.trace(model, example_input)
    exported_program = torch.jit.trace(model, example_input, check_trace=False)

    print(type(exported_program))

    #scripted_model = torch.jit.script(model)
    
    model_from_trace = ct.convert(
        exported_program,
        inputs=[ct.TensorType(shape=example_input.shape)],
        source='pytorch',
        minimum_deployment_target=ct.target.iOS17,  # Ensure we use latest CoreML features
    )
    
    print('Model converted to CoreML')
    model_from_trace.save(output_path)

def main():
    args = parse_args()
    
    # Load model and config
    model, config = load_model_and_config(
        args.model_type, 
        args.config_path, 
        args.checkpoint,
        args.use_stft,
        args.half
    )
    
    # Prepare audio
    audio = prepare_audio(args.example_audio, config, args.half)
    processed_audio = process_audio_chunk(audio, config, args.half)
    
    # Prepare model input
    if args.use_stft:
        model_input = compute_stft(processed_audio, config, args.half)
    else:
        model_input = processed_audio.unsqueeze(0).contiguous()
    
    # Convert to CoreML
    convert_to_coreml(model, model_input, args.output_path)

if __name__ == "__main__":
    main()
