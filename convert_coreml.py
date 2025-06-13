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
    model = model.half() if half else model

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            if hasattr(module, 'weight') and hasattr(module, 'bias'):
                if module.bias is not None:
                    if module.weight.dtype != module.bias.dtype:
                        print(f"{name}:")
                        print(f"  weight dtype: {module.weight.dtype}")
                        print(f"  bias dtype: {module.bias.dtype}")
                        print(f"  ❌ MISMATCH FOUND!")



    return model, config

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

    # with torch.no_grad():
    #     print(f"Example input shape: {example_input.shape}")
    #     original_output = model(example_input)
    #     print(f"Original model output stats:")
    #     print(f"  Shape: {original_output.shape}")
    #     print(f"  Min: {original_output.min():.6f}")
    #     print(f"  Max: {original_output.max():.6f}")
    #     print(f"  Mean: {original_output.mean():.6f}")
    #     print(f"  Non-zero elements: {torch.count_nonzero(original_output)}")
    
    # # Use script instead of trace for better handling of control flow
    # example_inputs = (torch.randn(1, 2, 242550, dtype=torch.float32),)
    # # example_inputs = (example_input,)
    # exported_program = torch.export.export(
    #     model, 
    #     example_inputs,
    # )

    # # executorch_program = to_edge_transform_and_lower(
    # #     exported_program,
    # #     partitioner = [CoreMLPartitioner()]
    # # ).to_executorch()

    # #exported_program = torch.jit.trace(model, example_input)
    exported_program = torch.jit.trace(model, example_input, check_trace=False)

    # # Test the traced model
    # with torch.no_grad():
    #     traced_output = exported_program(example_input)
    #     print(f"\nTraced model output stats:")
    #     print(f"  Shape: {traced_output.shape}")
    #     print(f"  Min: {traced_output.min():.6f}")
    #     print(f"  Max: {traced_output.max():.6f}")
    #     print(f"  Mean: {traced_output.mean():.6f}")
    #     print(f"  Non-zero elements: {torch.count_nonzero(traced_output)}")
        
    #     # Compare outputs
    #     diff = torch.abs(original_output - traced_output)
    #     print(f"  Max difference from original: {diff.max():.6f}")

    # print(type(exported_program))

    #scripted_model = torch.jit.script(model)

    try:
        scripted_model = torch.jit.script(model)
        scripted_output = scripted_model(example_input)
        print("Script method works - use this instead of trace")
    except Exception as e:
        print(f"Script method failed: {e}")
    
    model_from_trace = ct.convert(
        exported_program,
        inputs=[ct.TensorType(name="audio_input", 
                         shape=(1, 2, 242550),  # Your exact input shape
                         dtype=np.float32)],
        outputs=[ct.TensorType(name="separated_audio", 
                          dtype=np.float32)],
        # compute_units=ct.ComputeUnit.CPU_AND_GPU, 
        compute_precision=ct.precision.FLOAT32,
        source='pytorch',
        minimum_deployment_target=ct.target.iOS17,  # Ensure we use latest CoreML features
    )

    # Test with the same input you used for tracing
    test_input = example_input.numpy()  # Convert to numpy
    coreml_output = model_from_trace.predict({"audio_input": test_input})

    print("CoreML output stats:")
    output_array = coreml_output["separated_audio"]  # Adjust key name if different
    print(f"  Shape: {output_array.shape}")
    print(f"  Min: {output_array.min()}")
    print(f"  Max: {output_array.max()}")
    print(f"  Non-zero elements: {np.count_nonzero(output_array)}")
    
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
