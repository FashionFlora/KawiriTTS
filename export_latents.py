"""
export_latents.py - Export 25Hz latents from trained VAE2 model for flow matching training

This script extracts latent representations from the VAE2 encoder at 25Hz for all
audio files in the dataset, which can then be used to train a flow matching model.

Architecture:
- Loads trained VAE2 model
- Encodes mel spectrograms (100Hz) -> latents (25Hz, 128-dim)
- Saves latents as .pt files matching the original file structure

Usage:
    python export_latents.py -c configs/vae2_bottleneck.json -m logs/vae2_stage1 -o latents_25hz
    
    # Or with specific checkpoint:
    python export_latents.py -c configs/vae2_bottleneck.json --checkpoint logs/vae2_stage1/G_100000.pth -o latents_25hz
"""

import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

import utils
from models import SynthesizerTrn
from data_utils import TextAudioLoader, TextAudioCollate
from torch.utils.data import DataLoader
from text.symbols import symbols


def get_args():
    parser = argparse.ArgumentParser(description="Export 25Hz latents from VAE2 model")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to config JSON")
    parser.add_argument("-m", "--model_dir", type=str, default=None, help="Model directory (uses latest checkpoint)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Specific checkpoint path")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Output directory for latents")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "both"], help="Which split to process")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (1 recommended for variable length)")
    parser.add_argument("--save_format", type=str, default="pt", choices=["pt", "npy"], help="Output format")
    return parser.parse_args()


def load_model(hps, checkpoint_path, device):
    """Load the VAE2 model from checkpoint."""
    
    if (
        "use_mel_posterior_encoder" in hps.model.keys()
        and hps.model.use_mel_posterior_encoder == True
    ):
        posterior_channels = hps.data.n_mel_channels
    else:
        posterior_channels = hps.data.filter_length // 2 + 1
    
    # Create model
    model = SynthesizerTrn(
        len(symbols),
        posterior_channels,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model,
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle DDP wrapped checkpoints
    state_dict = checkpoint['model']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    
    return model


def export_latents(model, dataloader, output_dir, hps, save_format="pt", device="cuda"):
    """Extract and save latents for all samples."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Metadata storage
    metadata = []
    
    with torch.no_grad():
        for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths) in enumerate(tqdm(dataloader)):
            spec = spec.to(device)
            spec_lengths = spec_lengths.to(device)
            
            # Get mel if using mel posterior encoder
            if hps.model.use_mel_posterior_encoder or hps.data.use_mel_posterior_encoder:
                mel = spec
                mel_lengths = spec_lengths
            else:
                # Would need to convert spec to mel, but typically we use mel directly
                mel = spec
                mel_lengths = spec_lengths
            
            # Encode to latent
            latent, z_lengths, z_mask = model.encode_to_latent(mel, mel_lengths)
            
            # Save each sample in batch
            for i in range(latent.size(0)):
                sample_idx = batch_idx * dataloader.batch_size + i
                latent_len = z_lengths[i].item()
                
                # Get valid latent (remove padding)
                valid_latent = latent[i, :, :latent_len].cpu()
                
                # Create output filename
                output_filename = f"latent_{sample_idx:06d}"
                
                if save_format == "pt":
                    output_path = os.path.join(output_dir, f"{output_filename}.pt")
                    torch.save({
                        'latent': valid_latent,
                        'length': latent_len,
                        'mel_length': mel_lengths[i].item(),
                        'sample_idx': sample_idx,
                    }, output_path)
                else:  # npy
                    output_path = os.path.join(output_dir, f"{output_filename}.npy")
                    np.save(output_path, valid_latent.numpy())
                
                metadata.append({
                    'sample_idx': sample_idx,
                    'latent_file': os.path.basename(output_path),
                    'latent_length': latent_len,
                    'mel_length': mel_lengths[i].item(),
                    'latent_shape': list(valid_latent.shape),
                })
    
    # Save metadata
    metadata_path = os.path.join(output_dir, "metadata.pt")
    torch.save(metadata, metadata_path)
    print(f"Saved metadata to {metadata_path}")
    
    return metadata


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load config
    hps = utils.get_hparams_from_file(args.config)
    
    # Ensure VAE2 mode
    if not getattr(hps.model, 'use_vae2_bottleneck', False):
        raise ValueError("Config must have use_vae2_bottleneck=True")
    
    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    elif args.model_dir:
        checkpoint_path = utils.latest_checkpoint_path(args.model_dir, "G_*.pth")
    else:
        raise ValueError("Must specify either --model_dir or --checkpoint")
    
    print(f"Loading model from: {checkpoint_path}")
    model = load_model(hps, checkpoint_path, device)
    
    # Process requested splits
    splits_to_process = []
    if args.split == "both":
        splits_to_process = ["train", "val"]
    else:
        splits_to_process = [args.split]
    
    for split in splits_to_process:
        print(f"\nProcessing {split} split...")
        
        if split == "train":
            data_files = hps.data.training_files
        else:
            data_files = hps.data.validation_files
        
        # Create dataset and loader
        dataset = TextAudioLoader(data_files, hps.data)
        collate_fn = TextAudioCollate()
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn,
        )
        
        # Export latents
        output_dir = os.path.join(args.output_dir, split)
        metadata = export_latents(
            model, dataloader, output_dir, hps,
            save_format=args.save_format, device=device
        )
        
        print(f"Exported {len(metadata)} latents to {output_dir}")
        
        # Print summary statistics
        latent_lengths = [m['latent_length'] for m in metadata]
        print(f"  Latent length stats: min={min(latent_lengths)}, max={max(latent_lengths)}, "
              f"mean={np.mean(latent_lengths):.1f}")
        print(f"  Expected Hz: 25Hz (100Hz mel / 4x downsample)")
    
    print("\nDone! Latents are ready for flow matching training.")


if __name__ == "__main__":
    main()
