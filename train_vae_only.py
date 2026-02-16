"""
train_vae_only.py - Minimal VAE2 bottleneck + vocoder training script

No discriminator, no FM loss, no generator loss.
Just mel reconstruction + VAE KL divergence loss.

Usage:
    python train_vae_only.py -c configs/vae2_bottleneck.json -m logs/vae_only_test
"""

import os
"import shutil
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import tqdm
import argparse
import json

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate
from models import SynthesizerTrn, kl_divergence_loss
from preprocess.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from text.symbols import symbols

torch.backends.cudnn.benchmark = True


def get_hparams():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='JSON config file path')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Model output directory')
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        data = f.read()
    config = json.loads(data)
    
    hparams = utils.HParams(**config)
    hparams.model_dir = args.model
    return hparams, args


def main():
    hps, args = get_hparams()
    
    # Create model directory
    os.makedirs(hps.model_dir, exist_ok=True)
    
    # Save config (copy original config file)
    shutil.copy(args.config, os.path.join(hps.model_dir, "config.json"))
    
    # Setup logging
    logger = utils.get_logger(hps.model_dir)
    logger.info(hps)
    writer = SummaryWriter(log_dir=hps.model_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    torch.manual_seed(hps.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(hps.train.seed)
    
    # Get config values
    vae2_kl_weight = getattr(hps.model, 'vae2_kl_weight', 1e-4)
    
    if (
        "use_mel_posterior_encoder" in hps.model.keys()
        and hps.model.use_mel_posterior_encoder == True
    ):
        print("Using mel posterior encoder")
        posterior_channels = hps.data.n_mel_channels
        hps.data.use_mel_posterior_encoder = True
    else:
        print("Using lin posterior encoder")
        posterior_channels = hps.data.filter_length // 2 + 1
        hps.data.use_mel_posterior_encoder = False

    logger.info(f"VAE2 bottleneck mode:")
    logger.info(f"  - Latent dim: {getattr(hps.model, 'vae2_latent_dim', 128)}")
    logger.info(f"  - Downsample: {getattr(hps.model, 'vae2_downsample', 4)}x")
    logger.info(f"  - KL weight: {vae2_kl_weight}")
    logger.info(f"  - Mel loss weight: {hps.train.c_mel}")

    # Worker initialization function
    def worker_init_fn(worker_id):
        import numpy as np
        import random
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    # Data loading
    train_dataset = TextAudioLoader(hps.data.training_files, hps.data)
    collate_fn = TextAudioCollate()
    
    train_loader = DataLoader(
        train_dataset,
        num_workers=4,
        shuffle=True,
        batch_size=hps.train.batch_size,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        persistent_workers=True,
        prefetch_factor=2,
        worker_init_fn=worker_init_fn,
    )
    
    eval_dataset = TextAudioLoader(hps.data.validation_files, hps.data)
    eval_loader = DataLoader(
        eval_dataset,
        num_workers=2,
        shuffle=False,
        batch_size=hps.train.batch_size,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
        persistent_workers=True,
        prefetch_factor=2,
        worker_init_fn=worker_init_fn,
    )

    # Create model - only VAE2 encoder + upsampler + vocoder decoder
    net_g = SynthesizerTrn(
        len(symbols),
        posterior_channels,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model,
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in net_g.parameters())
    trainable_params = sum(p.numel() for p in net_g.parameters() if p.requires_grad)
    logger.info(f"Total params: {total_params:,}")
    logger.info(f"Trainable params: {trainable_params:,}")
    
    # Optimizer - only generator
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )

    # Load checkpoint if exists
    epoch_str = 1
    global_step = 0
    try:
        ckpt_path = utils.latest_checkpoint_path(hps.model_dir, "G_*.pth")
        if ckpt_path:
            _, _, _, epoch_str = utils.load_checkpoint(ckpt_path, net_g, optim_g)
            global_step = (epoch_str - 1) * len(train_loader)
            logger.info(f"Loaded checkpoint from epoch {epoch_str}")
    except Exception as e:
        logger.info(f"Starting from scratch: {e}")
        epoch_str = 1
        global_step = 0

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )

    scaler = GradScaler(enabled=hps.train.fp16_run)

    # Training loop
    logger.info("Starting training...")
    for epoch in range(epoch_str, hps.train.epochs + 1):
        net_g.train()
        
        loader = tqdm.tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths) in enumerate(loader):
            # Move to device
            spec = spec.to(device, non_blocking=True)
            spec_lengths = spec_lengths.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_lengths = y_lengths.to(device, non_blocking=True)
            x = x.to(device, non_blocking=True)
            x_lengths = x_lengths.to(device, non_blocking=True)

            with autocast(enabled=hps.train.fp16_run):
                # Forward pass
                (
                    y_hat,
                    l_length,  # Contains KL loss in VAE2 mode
                    attn,
                    ids_slice,
                    x_mask,
                    z_mask,
                    (z, z_p, m_p, logs_p, m_q, logs_q),
                    (hidden_x, logw, logw_),
                    (mag, phase)
                ) = net_g(x, x_lengths, spec, spec_lengths)

                # Get mel spectrograms
                if hps.model.use_mel_posterior_encoder or hps.data.use_mel_posterior_encoder:
                    mel = spec.float()
                else:
                    mel = spec_to_mel_torch(
                        spec.float(),
                        hps.data.filter_length,
                        hps.data.n_mel_channels,
                        hps.data.sampling_rate,
                        hps.data.mel_fmin,
                        hps.data.mel_fmax,
                    )
                
                # Slice mel to match output
                y_mel = commons.slice_segments(
                    mel, ids_slice, hps.train.segment_size // hps.data.hop_length
                )

                # Slice audio
                y = commons.slice_segments(
                    y, ids_slice * hps.data.hop_length, hps.train.segment_size
                )
                
                # Match y_hat length
                if y_hat.size(-1) > y.size(-1):
                    y_hat = y_hat[:, :, :y.size(-1)]
                elif y_hat.size(-1) < y.size(-1):
                    y_hat = F.pad(y_hat, (0, y.size(-1) - y_hat.size(-1)))

                # Compute mel of generated audio
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.squeeze(1).float(),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
                
                # Match mel shapes
                min_mel_len = min(y_mel.size(2), y_hat_mel.size(2))
                y_mel = y_mel[:, :, :min_mel_len]
                y_hat_mel = y_hat_mel[:, :, :min_mel_len]
                
                # Losses
                with autocast(enabled=False):
                    # Mel reconstruction loss
                    loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                    
                    # KL divergence loss (l_length contains KL in VAE2 mode)
                    loss_kl = l_length * vae2_kl_weight
                    
                    # Total loss - just mel + KL, nothing else
                    loss_total = loss_mel + loss_kl

            # Backprop
            optim_g.zero_grad()
            scaler.scale(loss_total).backward()
            scaler.unscale_(optim_g)
            grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
            scaler.step(optim_g)
            scaler.update()

            # Logging
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]["lr"]
                logger.info(
                    f"Epoch {epoch} [{100.0 * batch_idx / len(train_loader):.0f}%] | "
                    f"loss_mel: {loss_mel.item():.4f} | "
                    f"loss_kl: {loss_kl.item():.6f} | "
                    f"total: {loss_total.item():.4f} | "
                    f"lr: {lr:.6f}"
                )

                scalar_dict = {
                    "loss/total": loss_total,
                    "loss/mel": loss_mel,
                    "loss/kl": loss_kl,
                    "loss/kl_raw": l_length,
                    "learning_rate": lr,
                    "grad_norm": grad_norm_g,
                }
                
                image_dict = {
                    "slice/mel_org": utils.plot_spectrogram_to_numpy(
                        y_mel[0].data.cpu().numpy()
                    ),
                    "slice/mel_gen": utils.plot_spectrogram_to_numpy(
                        y_hat_mel[0].data.cpu().numpy()
                    ),
                    "all/mel": utils.plot_spectrogram_to_numpy(
                        mel[0].data.cpu().numpy()
                    ),
                }
                
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict,
                )

            # Evaluation & checkpoint
            if global_step % hps.train.eval_interval == 0:
                evaluate(hps, net_g, eval_loader, writer_eval, global_step, device)
                utils.save_checkpoint(
                    net_g,
                    optim_g,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(hps.model_dir, f"G_{global_step}.pth"),
                )
                logger.info(f"Saved checkpoint at step {global_step}")

            global_step += 1
            
            # Update tqdm
            loader.set_postfix({
                'mel': f'{loss_mel.item():.3f}',
                'kl': f'{loss_kl.item():.5f}',
            })

        scheduler_g.step()
        logger.info(f"====> Epoch {epoch} complete")


def evaluate(hps, generator, eval_loader, writer_eval, global_step, device):
    generator.eval()
    
    with torch.no_grad():
        for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths) in enumerate(eval_loader):
            spec = spec.to(device)
            spec_lengths = spec_lengths.to(device)
            y = y.to(device)
            y_lengths = y_lengths.to(device)

            spec = spec[:1]
            spec_lengths = spec_lengths[:1]
            y = y[:1]
            y_lengths = y_lengths[:1]
            break
        
        y_hat, mask, latent_info = generator.vocoder_infer(spec, spec_lengths, max_len=None)
        y_hat_lengths = mask.sum([1, 2]).long() * hps.data.hop_length

        if hps.model.use_mel_posterior_encoder or hps.data.use_mel_posterior_encoder:
            mel = spec
        else:
            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
        
        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1).float(),
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            hps.data.mel_fmin,
            hps.data.mel_fmax,
        )
    
    image_dict = {
        "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy()),
        "gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy()),
    }
    
    # Add latent visualization for VAE2 mode
    if len(latent_info) >= 2:
        latent_mean = latent_info[1]
        image_dict["gen/latent_25hz"] = utils.plot_spectrogram_to_numpy(
            latent_mean[0].cpu().numpy()
        )
    
    audio_dict = {
        "gen/audio": y_hat[0, :, :y_hat_lengths[0]],
        "gt/audio": y[0, :, :y_lengths[0]],
    }

    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate,
    )
    generator.train()


if __name__ == "__main__":
    main()
