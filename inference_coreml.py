# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'

import time
import librosa
import sys
import os
import glob
import torch
import math
import torch.nn.functional as F
import soundfile as sf
import numpy as np
from tqdm.auto import tqdm
import torch.nn as nn
import argparse
from omegaconf import DictConfig as ConfigDict

from typing import Dict, List, Tuple, Any, Union

from models.stft_utils import STFT_Process

# Using the embedded version of Python can also correctly import the utils module.
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from utils.audio_utils import normalize_audio, denormalize_audio, draw_spectrogram
from utils.settings import get_model_from_config, parse_args_inference
from utils.model_utils import demix
from utils.model_utils import prefer_target_instrument, apply_tta, load_start_checkpoint

import warnings

warnings.filterwarnings("ignore")

def _getWindowingArray(window_size: int, fade_size: int) -> torch.Tensor:
    """
    Generate a windowing array with a linear fade-in at the beginning and a fade-out at the end.

    This function creates a window of size `window_size` where the first `fade_size` elements
    linearly increase from 0 to 1 (fade-in) and the last `fade_size` elements linearly decrease
    from 1 to 0 (fade-out). The middle part of the window is filled with ones.

    Parameters:
    ----------
    window_size : int
        The total size of the window.
    fade_size : int
        The size of the fade-in and fade-out regions.

    Returns:
    -------
    torch.Tensor
        A tensor of shape (window_size,) containing the generated windowing array.

    Example:
    -------
    If `window_size=10` and `fade_size=3`, the output will be:
    tensor([0.0000, 0.5000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.5000, 0.0000])
    """

    fadein = torch.linspace(0, 1, fade_size)
    fadeout = torch.linspace(1, 0, fade_size)

    window = torch.ones(window_size)
    window[-fade_size:] = fadeout
    window[:fade_size] = fadein
    return window

def demix_coreml(
        config: ConfigDict,
        model: torch.nn.Module,
        mix: torch.Tensor,
        device: torch.device,
        model_type: str,
        pbar: bool = False
) -> Tuple[List[Dict[str, np.ndarray]], np.ndarray]:
    """
    Unified function for audio source separation with support for multiple processing modes.

    This function separates audio into its constituent sources using either a generic custom logic
    or a Demucs-specific logic. It supports batch processing and overlapping window-based chunking
    for efficient and artifact-free separation.

    Parameters:
    ----------
    config : ConfigDict
        Configuration object containing audio and inference settings.
    model : torch.nn.Module
        The trained model used for audio source separation.
    mix : torch.Tensor
        Input audio tensor with shape (channels, time).
    device : torch.device
        The computation device (CPU or CUDA).
    model_type : str, optional
        Processing mode:
            - "demucs" for logic specific to the Demucs model.
        Default is "generic".
    pbar : bool, optional
        If True, displays a progress bar during chunk processing. Default is False.

    Returns:
    -------
    Union[Dict[str, np.ndarray], np.ndarray]
        - A dictionary mapping target instruments to separated audio sources if multiple instruments are present.
        - A numpy array of the separated source if only one instrument is present.
    """

    # Ensure mix is a tensor and in the correct precision
    if not isinstance(mix, torch.Tensor):
        mix = torch.tensor(mix)
    
    # Move to device and convert to half precision
    # mix = mix.to(device).half()

    if model_type == 'htdemucs':
        mode = 'demucs'
    else:
        mode = 'generic'
    # Define processing parameters based on the mode
    if mode == 'demucs':
        # chunk_size = config.training.samplerate * config.training.segment
        chunk_size = config.audio.chunk_size
        num_instruments = len(config.training.instruments)
        num_overlap = config.inference.num_overlap
        step = chunk_size // num_overlap
    else:
        chunk_size = config.audio.chunk_size
        num_instruments = len(prefer_target_instrument(config))
        num_overlap = config.inference.num_overlap

        fade_size = chunk_size // 10
        step = chunk_size // num_overlap
        border = chunk_size - step
        length_init = mix.shape[-1]
        # Create windowing array in half precision
        windowing_array = _getWindowingArray(chunk_size, fade_size).to(device) # .half()
        
        # Add padding for generic mode to handle edge artifacts
        if length_init > 2 * border and border > 0:
            mix = nn.functional.pad(mix, (border, border), mode="reflect")

    batch_size = config.inference.batch_size

    use_amp = getattr(config.training, 'use_amp', True)

    with torch.cuda.amp.autocast(enabled=use_amp):
        with torch.inference_mode():
            # Initialize result and counter tensors in half precision
            req_shape = (num_instruments,) + mix.shape
            result = torch.zeros(req_shape, dtype=torch.float32, device=device)
            counter = torch.zeros(req_shape, dtype=torch.float32, device=device)

            stft_calc = STFT_Process(model_type='istft_C',
                                         n_fft=model.nfft, 
                                         win_length=model.nfft, 
                                         hop_len=model.hop_length, 
                                         max_frames=512,
                                         window_type='hann')
            
            # Move STFT_Process to the same device as the model
            stft_calc = stft_calc.to(device)

            i = 0
            batch_data = []
            batch_locations = []
            progress_bar = tqdm(
                total=mix.shape[1], desc="Processing audio chunks", leave=False
            ) if pbar else None

            while i < mix.shape[1]:
                # Extract chunk and apply padding if necessary
                part = mix[:, i:i + chunk_size].to(device)
                chunk_len = part.shape[-1]
                if mode == "generic" and chunk_len > chunk_size // 2:
                    pad_mode = "reflect"
                else:
                    pad_mode = "constant"
                part = nn.functional.pad(part, (0, chunk_size - chunk_len), mode=pad_mode, value=0)

                batch_data.append(part)

                print(part.shape)

                for ch in range(part.shape[0]):
                    channel_data = part[ch, :].cpu().numpy()
                    min_val = channel_data.min()
                    max_val = channel_data.max()
                    mean_val = channel_data.mean()
                    zeros = np.sum(channel_data == 0)
                    total = channel_data.size
                    print(f"[AudioInput] Channel {ch}: min={min_val:.7f}, max={max_val:.7f}, mean={mean_val:.7e}, zeros={zeros} out of {total}")




                batch_locations.append((i, chunk_len))
                i += step

                # Process batch if it's full or the end is reached
                if len(batch_data) >= batch_size or i >= mix.shape[1]:
                    arr = torch.stack(batch_data, dim=0)
                
                    print(arr.shape, arr.dtype)

                    ## torch.Size([2, 2, 485100])

                    # Print audio chunk statistics for each channel


                    x = model(arr)

                    if mode == "generic":
                        window = windowing_array.clone() # using clone() fixes the clicks at chunk edges when using batch_size=1
                        if i - step == 0:  # First audio chunk, no fadein
                            window[:fade_size] = 1
                        elif i >= mix.shape[1]:  # Last audio chunk, no fadeout
                            window[-fade_size:] = 1

                    for j, (start, seg_len) in enumerate(batch_locations):
                        if mode == "generic":
                            result[..., start:start + seg_len] += x[j, ..., :seg_len] * window[..., :seg_len]
                            counter[..., start:start + seg_len] += window[..., :seg_len]
                        else:
                            result[..., start:start + seg_len] += x[j, ..., :seg_len]
                            counter[..., start:start + seg_len] += 1.0

                    batch_data.clear()
                    batch_locations.clear()

                if progress_bar:
                    progress_bar.update(step)

            if progress_bar:
                progress_bar.close()

            # Compute final estimated sources
            estimated_sources = result / counter
            estimated_sources = estimated_sources.cpu().numpy().astype(np.float32)
            np.nan_to_num(estimated_sources, copy=False, nan=0.0)

            # Remove padding for generic mode
            if mode == "generic":
                if length_init > 2 * border and border > 0:
                    estimated_sources = estimated_sources[..., border:-border]

    # Return the result as a dictionary or a single array
    if mode == "demucs":
        instruments = config.training.instruments
    else:
        instruments = prefer_target_instrument(config)

    ret_data = {k: v for k, v in zip(instruments, estimated_sources)}

    if mode == "demucs" and num_instruments <= 1:
        return estimated_sources
    else:
        return ret_data


def run_folder(model, args, config, device, verbose: bool = False):
    """
    Process a folder of audio files for source separation.

    Parameters:
    ----------
    model : torch.nn.Module
        Pre-trained model for source separation.
    args : Namespace
        Arguments containing input folder, output folder, and processing options.
    config : Dict
        Configuration object with audio and inference settings.
    device : torch.device
        Device for model inference (CPU or CUDA).
    verbose : bool, optional
        If True, prints detailed information during processing. Default is False.
    """

    start_time = time.time()
    model.eval()

    print(f"Processing folder: {args.input_folder}")
    mixture_paths = sorted(glob.glob(os.path.join(args.input_folder, '*.*')))
    sample_rate = getattr(config.audio, 'sample_rate', 44100)

    print(f"Total files found: {len(mixture_paths)}. Using sample rate: {sample_rate}")

    instruments = prefer_target_instrument(config)[:]
    os.makedirs(args.store_dir, exist_ok=True)

    if not verbose:
        mixture_paths = tqdm(mixture_paths, desc="Total progress")

    if args.disable_detailed_pbar:
        detailed_pbar = False
    else:
        detailed_pbar = True

    for path in mixture_paths:
        print(f"Processing track: {path}")
        try:
            mix, sr = librosa.load(path, sr=sample_rate, mono=False)
        except Exception as e:
            print(f'Cannot read track: {format(path)}')
            print(f'Error message: {str(e)}')
            continue

        # If mono audio we must adjust it depending on model
        if len(mix.shape) == 1:
            mix = np.expand_dims(mix, axis=0)
            if 'num_channels' in config.audio:
                if config.audio['num_channels'] == 2:
                    print(f'Convert mono track to stereo...')
                    mix = np.concatenate([mix, mix], axis=0)

        mix_orig = mix.copy()
        if 'normalize' in config.inference:
            if config.inference['normalize'] is True:
                mix, norm_params = normalize_audio(mix)
        else:
            print(f"No normalization")

        # Convert input to tensor and ensure it's in half precision
        # mix = torch.from_numpy(mix).to(device).half()
        
        waveforms_orig = demix_coreml(config, model, mix, device, model_type=args.model_type, pbar=detailed_pbar)

        if args.use_tta:
            waveforms_orig = apply_tta(config, model, mix, waveforms_orig, device, args.model_type)

        if args.extract_instrumental:
            instr = 'vocals' if 'vocals' in instruments else instruments[0]
            # Convert waveforms_orig values to numpy before subtraction
            waveforms_orig_np = {k: v.cpu().numpy() for k, v in waveforms_orig.items()}
            waveforms_orig['instrumental'] = mix_orig - waveforms_orig_np[instr]
            if 'instrumental' not in instruments:
                instruments.append('instrumental')

        file_name = os.path.splitext(os.path.basename(path))[0]

        output_dir = os.path.join(args.store_dir, file_name)
        os.makedirs(output_dir, exist_ok=True)

        for instr in instruments:
            estimates = waveforms_orig[instr]
            if 'normalize' in config.inference:
                if config.inference['normalize'] is True:
                    estimates = denormalize_audio(estimates, norm_params)

            codec = 'flac' if getattr(args, 'flac_file', False) else 'wav'
            subtype = 'PCM_16' if args.flac_file and args.pcm_type == 'PCM_16' else 'FLOAT'

            output_path = os.path.join(output_dir, f"{instr}.{codec}")
            sf.write(output_path, estimates.T, sr, subtype=subtype)
            if args.draw_spectro > 0:
                output_img_path = os.path.join(output_dir, f"{instr}.jpg")
                draw_spectrogram(estimates.T, sr, args.draw_spectro, output_img_path)

    print(f"Elapsed time: {time.time() - start_time:.2f} seconds.")


def proc_folder(dict_args):
    args = parse_args_inference(dict_args)
    device = "cpu"
    if args.force_cpu:
        device = "cpu"
    elif torch.cuda.is_available():
        print('CUDA is available, use --force_cpu to disable it.')
        device = f'cuda:{args.device_ids[0]}' if isinstance(args.device_ids, list) else f'cuda:{args.device_ids}'
    elif torch.backends.mps.is_available():
        device = "mps"

    print("Using device: ", device)

    model_load_start_time = time.time()
    torch.backends.cudnn.benchmark = True

    model, config = get_model_from_config(args.model_type, args.config_path)

    if args.start_check_point != '':
        load_start_checkpoint(args, model, type_='inference')

    print("Instruments: {}".format(config.training.instruments))

    # Convert model to half precision before moving to device
    # model = model.half()

    # in case multiple CUDA GPUs are used and --device_ids arg is passed
    if isinstance(args.device_ids, list) and len(args.device_ids) > 1 and not args.force_cpu:
        model = nn.DataParallel(model, device_ids=args.device_ids)

    model = model.to(device)

    # Print model precision and parameter information
    print("\nModel Information:")
    print("-" * 50)
    
    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Check parameter precision
    param_dtypes = {}
    for name, param in model.named_parameters():
        dtype = str(param.dtype)
        if dtype not in param_dtypes:
            param_dtypes[dtype] = 0
        param_dtypes[dtype] += param.numel()
    
    print("\nParameter precision distribution:")
    for dtype, count in param_dtypes.items():
        percentage = (count / total_params) * 100
        print(f"{dtype}: {count:,} parameters ({percentage:.2f}%)")
    
    # Check if model is in eval mode
    print(f"\nModel in eval mode: {model.training == False}")
    
    # Check if model is using mixed precision
    if hasattr(model, 'half') and any(p.dtype == torch.float16 for p in model.parameters()):
        print("Model is using mixed precision (FP16)")
    elif hasattr(model, 'to') and any(p.dtype == torch.int8 for p in model.parameters()):
        print("Model is using INT8 quantization")
    else:
        print("Model is using full precision (FP32)")
    
    print("-" * 50 + "\n")

    print("Model load time: {:.2f} sec".format(time.time() - model_load_start_time))

    run_folder(model, args, config, device, verbose=True)


if __name__ == "__main__":
    proc_folder(None)
