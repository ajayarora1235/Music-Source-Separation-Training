import yaml
from utils.settings import ConfigDict
from models.mdx23c_tfc_tdf_v3 import TFC_TDF_net
import torch
import coremltools as ct
import numpy as np

config_path = "configs/config_mdx23c.yaml"
with open(config_path, 'r') as f:
  config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))

model = TFC_TDF_net(config).to("mps")

state_dict = torch.load("checkpoints/drumsep_5stems_mdx23c_jarredou.ckpt", map_location="cpu", weights_only=True)
model.load_state_dict(state_dict)

dummy_input = torch.randn((2, 2, 523776), dtype=torch.float32, device="mps")


# Trace only the PyTorch model (not the wrapper class)
model.eval()
with torch.no_grad():
    # Test forward pass first to ensure it works
    test_output = model(dummy_input)
    print(f"Model output shape: {test_output.shape}")

    # Now export the model
    traced_model = torch.jit.trace(model, dummy_input, check_trace=False)

# Convert to CoreML

model_from_trace = ct.convert(
    traced_model,
    inputs=[ct.TensorType(name="model_input",
                          shape=(2, 2, 523776),
                          dtype=np.float32)],
    outputs=[ct.TensorType(name="spec_output")],
    minimum_deployment_target=ct.target.iOS18,
    compute_precision=ct.precision.FLOAT32,
    compute_units=ct.ComputeUnit.CPU_AND_GPU,
    convert_to="mlprogram",
    source='pytorch',
)
model_from_trace.save("drumsep_full.mlpackage")