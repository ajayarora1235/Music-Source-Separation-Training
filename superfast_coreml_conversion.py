import yaml
from utils.settings import ConfigDict
from models.mdx23c_tfc_tdf_v3 import TFC_TDF_net
import torch
import coremltools as ct
import numpy as np
import argparse
from utils.model_utils import prefer_target_instrument, apply_tta, load_start_checkpoint
from utils.settings import get_model_from_config

def convert_to_coreml(model_type, config_path, start_check_point):
    model, config = get_model_from_config(model_type, config_path)

    if start_check_point != '':
        args = argparse.Namespace(start_check_point=start_check_point, model_type=model_type, config_path=config_path)
        load_start_checkpoint(args, model, type_='inference')

    batch_size = config.inference.batch_size
    chunk_size = config.audio.chunk_size
    num_instruments = len(prefer_target_instrument(config))

    dummy_input = torch.randn((batch_size, 2, chunk_size), dtype=torch.float32)


    # Trace only the PyTorch model (not the wrapper class)
    model.eval()
    with torch.no_grad():
        # Test forward pass first to ensure it works
        test_output = model(dummy_input)
        if model_type != 'htdemucs':
          print(f"Model output shape: {test_output.shape}")

        # Now export the model
        traced_model = torch.jit.trace(model, dummy_input, check_trace=False)

    # Convert to CoreML

    outputs = []
    if model_type == 'htdemucs':
      outputs = [ct.TensorType(name="spec_output", dtype=np.float32), ct.TensorType(name="mask_output", dtype=np.float32)]
    else:
      outputs = [ct.TensorType(name="spec_output", dtype=np.float32)]

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
    model_from_trace.save(f"{model_type}.mlpackage")

def main():
    passed_tests = ['htdemucs','mdx23c', 'scnet', 'bs_roformer_apple_coreml', 'bs_roformer']
    model_types = ['htdemucs', 'mdx23c', 'mel_band_roformer', 'scnet', 'bs_roformer_apple_coreml', 'bs_roformer']
    configs = ['configs/config_htdemucs_4stems.yaml', 'configs/config_mdx23c.yaml', 'configs/KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml', 'configs/config_musdb18_scnet_large_starrytong.yaml', 'configs/config_v1_apple_model.yaml', 'configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml']
    checkpoints = ['checkpoints/demucs.th', 'checkpoints/drumsep_5stems_mdx23c_jarredou.ckpt', 'checkpoints/MelBandRoformer.ckpt', 'checkpoints/SCNet-large_starrytong_fixed.ckpt', 'checkpoints/logic_roformer-fp16.pt', 'checkpoints/model_bs_roformer_ep_317_sdr_12.9755.ckpt']

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