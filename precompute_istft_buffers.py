#!/usr/bin/env python3
"""
Script to precompute ISTFT buffers for existing HTDemucs checkpoints.
This will load your existing checkpoint, compute the ISTFT buffers once,
and save it back with the buffers included for faster future loading.

UNUSED
"""

import torch
import argparse
from pathlib import Path

def precompute_istft_buffers(checkpoint_path, config_path, output_path=None):
    """
    Load checkpoint, compute ISTFT buffers, and save enhanced checkpoint.
    
    Args:
        checkpoint_path: Path to existing checkpoint
        output_path: Path to save enhanced checkpoint (defaults to overwriting original)
    """
    if output_path is None:
        output_path = checkpoint_path
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Import model class
    from utils.settings import get_model_from_config
    
    # Get model and config
    model, config = get_model_from_config('htdemucs', config_path)
    
    # Load the state dict
    model.load_state_dict(checkpoint['state'])
    
    print("Computing ISTFT buffers...")
    
    # Create a dummy input to trigger ISTFT computation
    # This should match your typical input dimensions

    dummy_z = torch.randn(16, 2049, 478, dtype=torch.complex64)  # Shape matches STFT output with n_fft=4096, hop_length=1024, length=488448
    
    # Trigger the ISTFT computation (this will compute and register the buffers)
    try:
        with torch.no_grad():
            _ = model._istft(dummy_z, n_fft=4096, hop_length=1024, length=488448)
        print("✅ ISTFT buffers computed successfully!")
    except Exception as e:
        print(f"❌ Error computing ISTFT buffers: {e}")
        return False
    
    # Update the checkpoint with new state dict (now includes ISTFT buffers)
    

    # Verify the buffers are in state_dict
    state_dict = model.state_dict()
    istft_buffers = [key for key in state_dict.keys() if key.startswith('istft_')]
    if not istft_buffers:
        print("❌ No ISTFT buffers found in state dict!")
        return False
    print(f"Found ISTFT buffers in state dict: {istft_buffers}")

    checkpoint['state'] = model.state_dict()
    print(f"Saving enhanced checkpoint to: {output_path}")
    
    # Save the enhanced checkpoint
    torch.save(checkpoint, output_path)
    
    print("✅ Enhanced checkpoint saved successfully!")
    
    # Verify the buffers are present
    enhanced_state_keys = set(checkpoint['state'].keys())
    istft_buffers = [key for key in enhanced_state_keys if key.startswith('istft_')]
    print(f"Added ISTFT buffers: {istft_buffers}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Precompute ISTFT buffers for HTDemucs checkpoint')
    parser.add_argument('checkpoint', type=str, help='Path to existing checkpoint file')
    parser.add_argument('config', type=str, help='Path to existing config file')
    parser.add_argument('--output', '-o', type=str, help='Output path (defaults to overwriting input)')
    parser.add_argument('--backup', '-b', action='store_true', help='Create backup of original file')
    
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint file not found: {checkpoint_path}")
        return
    
    # Create backup if requested
    if args.backup:
        backup_path = checkpoint_path.with_suffix(checkpoint_path.suffix + '.backup')
        print(f"Creating backup: {backup_path}")
        torch.save(torch.load(checkpoint_path, map_location='cpu'), backup_path)
    
    # Determine output path
    output_path = args.output if args.output else str(checkpoint_path)
    
    # Precompute buffers
    success = precompute_istft_buffers(str(checkpoint_path), args.config, output_path)
    
    if success:
        print(f"\n🎉 Success! Your checkpoint now includes precomputed ISTFT buffers.")
        print(f"Future model loading will be faster and won't require on-the-fly computation.")
    else:
        print(f"\n❌ Failed to precompute ISTFT buffers.")

if __name__ == "__main__":
    main() 