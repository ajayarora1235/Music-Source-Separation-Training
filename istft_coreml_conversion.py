import sys
import os
import torch
import numpy as np
import argparse
from einops import rearrange
import coremltools as ct

# Using the embedded version of Python can also correctly import the utils module.
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from utils.settings import get_model_from_config
from utils.model_utils import prefer_target_instrument, load_start_checkpoint
from STFT_Process import STFT_Process

def convert_to_coreml(model_type, config_path, start_check_point):
    model, config = get_model_from_config(model_type, config_path)

    if start_check_point != '':
        args = argparse.Namespace(start_check_point=start_check_point, model_type=model_type, config_path=config_path)
        load_start_checkpoint(args, model, type_='inference')

    batch_size = config.inference.batch_size
    chunk_size = config.audio.chunk_size

    dummy_input = torch.randn((batch_size, 2, chunk_size), dtype=torch.float32)
    device = torch.device('cpu')  # Use CPU for CoreML conversion

    # Get model output to understand the shape for istft_process
    model.eval()
    with torch.no_grad():
        # Test forward pass first to ensure it works
        test_output = model(dummy_input)
        if model_type != 'htdemucs':
            print(f"Model output shape: {test_output.shape}")

        # Pre-create ISTFT processor outside the traced model to avoid complex tensor issues
        # Extract STFT parameters based on model type
        if model_type == 'scnet':
            n_fft = config.model.nfft
            hop_length = config.model.hop_size
            normalized = config.model.normalized
        elif model_type == 'htdemucs':
            n_fft = config.htdemucs.nfft
            hop_length = config.audio.hop_length
            normalized = True
        else:
            # All other models (bs_roformer, mdx23c, mel_band_roformer, etc.)
            n_fft = config.audio.n_fft
            hop_length = config.audio.hop_length
            normalized = True
        
        custom_istft = STFT_Process(model_type='istft_C', n_fft=n_fft, hop_len=hop_length, normalized=normalized).to(device)

        # Create a wrapper that combines model inference and istft_process
        class ModelWithISTFT(torch.nn.Module):
            def __init__(self, model, config, model_type, arr_shape, istft_processor):
                super().__init__()
                self.model = model
                self.config = config
                self.model_type = model_type
                self.arr_shape = arr_shape
                self.istft_processor = istft_processor
                
            def forward(self, x):
                # Get model output
                model_output = self.model(x)
                # Apply istft_process with pre-created ISTFT processor
                return self.istft_forward(model_output, torch.zeros(self.arr_shape))
            
            def istft_forward(self, x, arr):
                if self.model_type == 'htdemucs':
                    # Special handling for htdemucs
                    z_real = x[0][..., 0]
                    z_imag = x[0][..., 1]
                    *other, freqs, frames = z_real.shape
                    z_real = z_real.view(-1, freqs, frames)
                    z_imag = z_imag.view(-1, freqs, frames)
                    correct_x = self.istft_processor.istft_C_forward(z_real, z_imag)
                    correct_x = correct_x.squeeze()
                    correct_x = correct_x.view(*other, -1)
                    return correct_x + x[1]
                else:
                    # Standard processing for all other models
                    num_instruments = len(prefer_target_instrument(self.config))
                    recon_audio = self.istft_processor.istft_C_forward(x[..., 0], x[..., 1])
                    recon_audio = recon_audio.squeeze()
                    recon_audio = rearrange(recon_audio, "(b n s) t -> b n s t", s=2, n=num_instruments)
                    return recon_audio

        # Create wrapper instance
        wrapped_model = ModelWithISTFT(model, config, model_type, dummy_input.shape, custom_istft)
        wrapped_model.eval()
        
        # Test the wrapped model
        test_wrapped_output = wrapped_model(dummy_input)
        print(f"Wrapped model output shape: {test_wrapped_output.shape}")

        # Now export the wrapped model (model + istft_process)
        traced_model = torch.jit.trace(wrapped_model, dummy_input, check_trace=False)

    # Convert to CoreML

    outputs = []
    if model_type == 'htdemucs':
      outputs = [ct.TensorType(name="audio_output", dtype=np.float32)]
    else:
      outputs = [ct.TensorType(name="audio_output", dtype=np.float32)]

    model_from_trace = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="model_input",
                              shape=(batch_size, 2, chunk_size),
                              dtype=np.float32)],
        outputs=outputs,
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
        convert_to="mlprogram",
        source='pytorch',
    )
    model_from_trace.save(f"{model_type}_istft.mlpackage")

def main():
    passed_tests = []
    model_types = ['mdx23c', 'mel_band_roformer', 'scnet', 'bs_roformer_apple_coreml', 'bs_roformer']
    configs = ['configs/config_mdx23c.yaml', 'configs/KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml', 'configs/config_musdb18_scnet_large_starrytong.yaml', 'configs/config_v1_apple_model.yaml', 'configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml']
    checkpoints = ['checkpoints/drumsep_5stems_mdx23c_jarredou.ckpt', 'checkpoints/MelBandRoformer.ckpt', 'checkpoints/SCNet-large_starrytong_fixed.ckpt', 'checkpoints/logic_roformer-fp16.pt', 'checkpoints/model_bs_roformer_ep_317_sdr_12.9755.ckpt']

    for model_type, config_path, checkpoint in zip(model_types, configs, checkpoints):
        if model_type in passed_tests:
            print(f"Skipping {model_type} as it has already passed tests.")
            continue
        try:
            convert_to_coreml(model_type, config_path, checkpoint)
        except Exception as e:
           print(f"Error converting {model_type} to CoreML: {e}")

if __name__ == "__main__":
    main()