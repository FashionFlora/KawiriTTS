"""
Test script to check if VAE has collapsed.
Checks:
1. Latent statistics from real data (should use the full latent space)
2. Prior sampling diversity (random N(0,1) latents should produce varied audio)
3. Reconstruction quality vs prior sampling comparison
"""

import os
import torch
import torchaudio
import numpy as np
from pathlib import Path
import utils
from models import SynthesizerTrn
from text.symbols import symbols
import scipy.io.wavfile as wavfile


def save_audio(path, audio, sample_rate):
    """Save audio using scipy (more reliable than torchaudio)."""
    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy()
    if audio.ndim == 2:
        audio = audio.squeeze(0)
    # Normalize to int16 range
    audio = np.clip(audio, -1, 1)
    audio_int16 = (audio * 32767).astype(np.int16)
    wavfile.write(path, sample_rate, audio_int16)


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
    # Handle different checkpoint formats
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'models' in checkpoint:
        state_dict = checkpoint['models']['model']
    else:
        state_dict = checkpoint
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load config  
    config_path = "configs/vae2_bottleneck.json"
    hps = utils.get_hparams_from_file(config_path)
    
    # Find latest checkpoint
    model_dir = "logs/vae_normal"
    checkpoint_path = utils.latest_checkpoint_path(model_dir, "G_*.pth")
    print(f"Loading checkpoint: {checkpoint_path}")
    
    model = load_model(hps, checkpoint_path, device)
    
    # Get some validation mels
    mel_dir = Path("../dump_100/val/mels")
    mel_files = list(mel_dir.glob("*.pt"))[:10]
    
    print("\n" + "="*60)
    print("TEST 1: Encoding real mels - checking latent statistics")
    print("="*60)
    
    all_means = []
    all_logvars = []
    all_latents = []
    
    for mel_path in mel_files:
        mel = torch.load(mel_path)
        if isinstance(mel, dict):
            mel = mel.get('mel', mel.get('spectrogram', mel))
        mel = mel.float().to(device)
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        
        mel_lengths = torch.tensor([mel.size(2)], device=device)
        
        with torch.no_grad():
            # Get the encoder output with mean and log_var
            z_latent, mean, log_var, z_mask, z_lengths = model.vae2_encoder(mel, mel_lengths, g=None)
        
        # Flatten time dimension for statistics [B, C, T] -> [B*T, C]
        all_means.append(mean.squeeze(0).transpose(0, 1).cpu())  # [T, C]
        all_logvars.append(log_var.squeeze(0).transpose(0, 1).cpu())  # [T, C]
        all_latents.append(z_latent.squeeze(0).transpose(0, 1).cpu())  # [T, C]
    
    # Concatenate all latent statistics along time
    all_means = torch.cat(all_means, dim=0)  # [total_T, latent_dim]
    all_logvars = torch.cat(all_logvars, dim=0)
    all_latents = torch.cat(all_latents, dim=0)
    
    print(f"\nLatent mean statistics:")
    print(f"  Mean of means: {all_means.mean().item():.4f}")
    print(f"  Std of means: {all_means.std().item():.4f}")
    print(f"  Min/Max of means: [{all_means.min().item():.4f}, {all_means.max().item():.4f}]")
    
    print(f"\nLatent log_var statistics:")
    print(f"  Mean of log_var: {all_logvars.mean().item():.4f}")
    print(f"  Std of log_var: {all_logvars.std().item():.4f}")
    print(f"  -> Implied average std: {(0.5 * all_logvars).exp().mean().item():.4f}")
    
    print(f"\nSampled latent (z) statistics:")
    print(f"  Mean: {all_latents.mean().item():.4f}")
    print(f"  Std: {all_latents.std().item():.4f}")
    
    # Check for collapse indicators
    print("\n" + "="*60)
    print("COLLAPSE INDICATORS:")
    print("="*60)
    
    # Check if latent variance is too low (posterior collapse)
    avg_latent_std = (0.5 * all_logvars).exp().mean().item()
    if avg_latent_std > 0.9:
        print(f"[WARNING] Posterior variance close to prior (std={avg_latent_std:.4f} ~= 1.0)")
        print("  -> This suggests posterior collapse (VAE ignoring latent code)")
    elif avg_latent_std < 0.1:
        print(f"[WARNING] Posterior variance very small (std={avg_latent_std:.4f})")
        print("  -> KL term might be too weak, not regularizing to prior")
    else:
        print(f"[OK] Posterior std = {avg_latent_std:.4f} (reasonable range)")
    
    # Check latent utilization
    latent_dim_std = all_latents.std(dim=0)  # variance per latent dimension [latent_dim]
    active_dims = (latent_dim_std > 0.1).sum().item()
    total_dims = latent_dim_std.size(0)
    print(f"\nActive latent dimensions: {active_dims}/{total_dims}")
    if active_dims < total_dims * 0.5:
        print(f"[WARNING] Only {active_dims/total_dims*100:.1f}% of latent dims are active!")
    else:
        print(f"[OK] {active_dims/total_dims*100:.1f}% of latent dimensions are active")
    
    print("\n" + "="*60)
    print("TEST 2: Sampling from prior N(0,1)")
    print("="*60)
    
    # Sample multiple random latents and decode
    n_samples = 5
    latent_dim = hps.model.vae2_latent_dim
    latent_length = 50  # ~2 seconds at 25Hz
    
    os.makedirs("test_outputs", exist_ok=True)
    
    audios = []
    for i in range(n_samples):
        # Sample from standard normal prior
        z_prior = torch.randn(1, latent_dim, latent_length, device=device)
        z_lengths = torch.tensor([latent_length], device=device)
        
        with torch.no_grad():
            audio, _ = model.decode_from_latent(z_prior, z_lengths)
        
        audios.append(audio.cpu())
        
        # Save audio
        out_path = f"test_outputs/prior_sample_{i}.wav"
        save_audio(out_path, audio.squeeze(0).cpu(), hps.data.sampling_rate)
        print(f"Saved {out_path}")
    
    # Check diversity of prior samples
    audios_stacked = torch.stack([a.squeeze() for a in audios])
    
    # Compute pairwise distances between samples
    print("\nPairwise L2 distances between prior samples (higher = more diverse):")
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            # Normalize lengths
            min_len = min(audios_stacked[i].size(-1), audios_stacked[j].size(-1))
            dist = (audios_stacked[i, :min_len] - audios_stacked[j, :min_len]).pow(2).mean().sqrt().item()
            print(f"  Sample {i} vs {j}: {dist:.4f}")
    
    print("\n" + "="*60)
    print("TEST 3: Reconstruction vs Prior comparison")
    print("="*60)
    
    # Pick one mel and compare reconstruction with prior sample
    mel = torch.load(mel_files[0])
    if isinstance(mel, dict):
        mel = mel.get('mel', mel.get('spectrogram', mel))
    mel = mel.float().to(device)
    if mel.dim() == 2:
        mel = mel.unsqueeze(0)
    
    mel_lengths = torch.tensor([mel.size(2)], device=device)
    
    with torch.no_grad():
        # Reconstruction
        audio_recon, _, _ = model.vocoder_infer(mel, mel_lengths)
        
        # Also encode and check what latent looks like
        z_latent, mean, log_var, z_mask, z_lengths = model.vae2_encoder(mel, mel_lengths, g=None)
    
    save_audio("test_outputs/reconstruction.wav", audio_recon.squeeze(0).cpu(), hps.data.sampling_rate)
    print("Saved test_outputs/reconstruction.wav")
    
    print(f"\nEncoded latent shape: {z_latent.shape}")
    print(f"Encoded latent mean: {mean.mean().item():.4f}, std: {mean.std().item():.4f}")
    print(f"Encoded latent log_var mean: {log_var.mean().item():.4f}")
    
    # Compare reconstruction audio statistics with prior samples
    recon_energy = audio_recon.pow(2).mean().item()
    prior_energy = audios_stacked.pow(2).mean().item()
    print(f"\nAudio energy - Reconstruction: {recon_energy:.6f}")
    print(f"Audio energy - Prior samples: {prior_energy:.6f}")
    
    if prior_energy < recon_energy * 0.01:
        print("[WARNING] Prior samples have very low energy compared to reconstruction!")
        print("  -> Decoder may not be responding to prior samples")
    elif prior_energy > recon_energy * 100:
        print("[WARNING] Prior samples have much higher energy than reconstruction!")
    else:
        print("[OK] Prior and reconstruction energy are in similar range")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Listen to the generated files in test_outputs/ to manually verify:")
    print(f"  - prior_sample_*.wav should sound like varied speech/audio")
    print(f"  - reconstruction.wav should sound like the original input")
    print(f"\nIf prior samples all sound identical or like noise, the VAE has collapsed.")
    print(f"If reconstruction sounds good but prior samples don't, the decoder")
    print(f"hasn't learned to use the latent space properly.\n")


if __name__ == "__main__":
    main()
