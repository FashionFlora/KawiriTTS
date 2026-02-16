"""
inference_vae2.py - Inference script for VAE2 bottleneck model

This script provides functions to:
1. Generate audio from mel spectrograms (mel -> latent -> audio)
2. Generate audio from 25Hz latents directly (latent -> audio)
3. Extract 25Hz latents from mel spectrograms (mel -> latent)

Usage:
    # From mel to audio
    python inference_vae2.py -c configs/vae2_bottleneck.json -m logs/vae2_stage1 \
        --mel path/to/mel.pt --output output.wav
    
    # From latent to audio
    python inference_vae2.py -c configs/vae2_bottleneck.json -m logs/vae2_stage1 \
        --latent path/to/latent.pt --output output.wav
    
    # Extract latent from mel
    python inference_vae2.py -c configs/vae2_bottleneck.json -m logs/vae2_stage1 \
        --mel path/to/mel.pt --output_latent latent.pt
"""

import os
import argparse
import torch
import torchaudio
import numpy as np
from pathlib import Path

import utils
from models import SynthesizerTrn
from text.symbols import symbols


def get_args():
    parser = argparse.ArgumentParser(description="VAE2 model inference")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to config JSON")
    parser.add_argument("-m", "--model_dir", type=str, default=None, help="Model directory")
    parser.add_argument("--checkpoint", type=str, default=None, help="Specific checkpoint path")
    
    # Input options (mutually exclusive)
    parser.add_argument("--mel", type=str, default=None, help="Input mel spectrogram (.pt or .npy)")
    parser.add_argument("--latent", type=str, default=None, help="Input 25Hz latent (.pt or .npy)")
    parser.add_argument("--audio", type=str, default=None, help="Input audio file (will compute mel)")
    
    # Output options
    parser.add_argument("--output", type=str, default=None, help="Output audio file (.wav)")
    parser.add_argument("--output_latent", type=str, default=None, help="Output latent file (.pt)")
    
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
    
    model = SynthesizerTrn(
        len(symbols),
        posterior_channels,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model,
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
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


def load_mel(mel_path, device):
    """Load mel spectrogram from file."""
    if mel_path.endswith('.npy'):
        mel = torch.from_numpy(np.load(mel_path)).float()
    else:  # .pt
        data = torch.load(mel_path)
        if isinstance(data, dict):
            mel = data.get('mel', data.get('spectrogram', data))
        else:
            mel = data
        mel = mel.float()
    
    if mel.dim() == 2:
        mel = mel.unsqueeze(0)  # Add batch dimension
    
    return mel.to(device)


def load_latent(latent_path, device):
    """Load latent from file."""
    if latent_path.endswith('.npy'):
        latent = torch.from_numpy(np.load(latent_path)).float()
    else:  # .pt
        data = torch.load(latent_path)
        if isinstance(data, dict):
            latent = data.get('latent', data)
        else:
            latent = data
        latent = latent.float()
    
    if latent.dim() == 2:
        latent = latent.unsqueeze(0)  # Add batch dimension
    
    return latent.to(device)


def mel_to_audio(model, mel, device):
    """Generate audio from mel spectrogram via VAE2 bottleneck."""
    mel_lengths = torch.tensor([mel.size(2)], device=device)
    
    with torch.no_grad():
        audio, mask, _ = model.vocoder_infer(mel, mel_lengths)
    
    return audio.squeeze().cpu()


def latent_to_audio(model, latent, device):
    """Generate audio from 25Hz latent."""
    latent_lengths = torch.tensor([latent.size(2)], device=device)
    
    with torch.no_grad():
        audio, mask = model.decode_from_latent(latent, latent_lengths)
    
    return audio.squeeze().cpu()


def mel_to_latent(model, mel, device):
    """Extract 25Hz latent from mel spectrogram."""
    mel_lengths = torch.tensor([mel.size(2)], device=device)
    
    with torch.no_grad():
        latent, z_lengths, z_mask = model.encode_to_latent(mel, mel_lengths)
    
    return latent.squeeze().cpu(), z_lengths.item()


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
    
    # Process based on input type
    if args.mel:
        print(f"Loading mel from: {args.mel}")
        mel = load_mel(args.mel, device)
        print(f"Mel shape: {mel.shape}")
        
        if args.output_latent:
            # Extract and save latent
            latent, latent_len = mel_to_latent(model, mel, device)
            print(f"Extracted latent shape: {latent.shape} (length: {latent_len})")
            torch.save({
                'latent': latent,
                'length': latent_len,
                'mel_length': mel.size(2),
            }, args.output_latent)
            print(f"Saved latent to: {args.output_latent}")
        
        if args.output:
            # Generate audio
            audio = mel_to_audio(model, mel, device)
            print(f"Generated audio shape: {audio.shape}")
            torchaudio.save(args.output, audio.unsqueeze(0), hps.data.sampling_rate)
            print(f"Saved audio to: {args.output}")
    
    elif args.latent:
        print(f"Loading latent from: {args.latent}")
        latent = load_latent(args.latent, device)
        print(f"Latent shape: {latent.shape}")
        
        if args.output:
            # Generate audio from latent
            audio = latent_to_audio(model, latent, device)
            print(f"Generated audio shape: {audio.shape}")
            torchaudio.save(args.output, audio.unsqueeze(0), hps.data.sampling_rate)
            print(f"Saved audio to: {args.output}")
        else:
            print("Warning: No output specified. Use --output to save audio.")
    
    elif args.audio:
        # For future implementation: compute mel from audio
        raise NotImplementedError("Audio input not yet implemented. Please provide pre-computed mel.")
    
    else:
        raise ValueError("Must specify one of: --mel, --latent, or --audio")


if __name__ == "__main__":
    main()
