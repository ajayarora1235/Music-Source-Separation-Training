import torch
import coremltools as ct
from typing import Tuple, Any
import argparse
import numpy as np
from utils.settings import get_model_from_config, parse_args_inference
from utils.model_utils import demix
from utils.model_utils import prefer_target_instrument, apply_tta, load_start_checkpoint


NFFT=2048
WIN_LENGTH=2048
HOP_LENGTH=1024
INPUT_LENGTH=117760

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Convert model to CoreML format')
    parser.add_argument('--config_path', type=str, required=True,
                      help='Path to the model config file')
    parser.add_argument('--start_check_point', type=str, required=True,
                      help='Path to the model checkpoint')
    parser.add_argument('--model_type', type=str, required=True,
                      help='Type of model (e.g., bs_roformer)')
    parser.add_argument('--output_path', type=str, default='model.mlpackage',
                      help='Path to save the CoreML model')
    parser.add_argument('--batch_size', type=int, default=1,
                      help='Batch size for model input (default: 1)')
    return parser.parse_args()

def load_model_and_config(model_type: str, config_path: str, checkpoint_path: str) -> Tuple[torch.nn.Module, Any]:
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


        ## filter out padding_zero
        state_dict = {k: v for k, v in state_dict.items() 
                     if 'padding_zero' not in k}
        
        # conv_tr_keys = [k for k in state_dict.keys() if 'conv_tr' in k]
        # for key in conv_tr_keys:
        #     del state_dict[key]
    else:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)

    
    filtered_state_dict = state_dict
    print('found number of keys: ', len(filtered_state_dict.keys()))
    
    model.load_state_dict(filtered_state_dict, strict=False)

    return model, config

def main():
    args = {
        'model_type': 'bs_roformer_apple_coreml',
        'config_path': 'configs/config_v1_apple_model.yaml',
        'start_check_point': 'checkpoints/logic_roformer-fp16.pt',
        'output_path': 'roformer_batched.mlpackage',
        'batch_size': 1,
    }

    args = argparse.Namespace(**args)
    model, config = get_model_from_config(args.model_type, args.config_path)

    if args.start_check_point != '':
        load_start_checkpoint(args, model, type_='inference')

    # Remove model.half() as CoreML STFT operations require fp32
    model = model.half()

    model.eval()  # Ensure model is in eval mode
    model_output_path = args.output_path 
    print("model_output_path: ", model_output_path)

    # dummy model input is z_real, z_imag, mix
    bins = int(INPUT_LENGTH / HOP_LENGTH) + 1
    batch_size = 2

    dummy_demucs_input = (
                torch.randn((2*batch_size, NFFT // 2, bins), dtype=torch.float32),
                torch.randn((2*batch_size, NFFT // 2, bins), dtype=torch.float32),
                torch.randn((batch_size, 2, INPUT_LENGTH), dtype=torch.float32))
    
    dummy_roformer_input = (
        torch.randn((batch_size, 2, INPUT_LENGTH), dtype=torch.float32),
    )
    
    # model output is xt, zout

    exported_program = torch.jit.trace(model, dummy_roformer_input, check_trace=False)
    model_from_trace = ct.convert(
        exported_program,
        inputs=[ct.TensorType(name="mix", 
                         shape=dummy_roformer_input[0].shape,  # Use batch_size parameter
                         dtype=np.float32)],
        outputs=[ct.TensorType(name="stft_output")],
        minimum_deployment_target=ct.target.iOS18,
        compute_units=ct.ComputeUnit.CPU_AND_GPU, 
        convert_to="mlprogram",
        source='pytorch',
    )
    model_from_trace.save(model_output_path)

if __name__ == "__main__":
    main()