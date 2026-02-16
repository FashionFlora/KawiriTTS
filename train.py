import os
import itertools
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
import tqdm

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, DistributedBucketSampler
from models import (
    SynthesizerTrn,
    MultiPeriodDiscriminator,
    MultiScaleSubbandCQTDiscriminator,
    kl_divergence_loss,
)
from losses import generator_loss, discriminator_loss, feature_loss, phase_loss
from preprocess.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from text.symbols import symbols

torch.backends.cudnn.benchmark = True
global_step = 0


def worker_init_fn(worker_id):
    """Worker initialization for DataLoader multiprocessing."""
    import numpy as np
    import random
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main():
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."

    n_gpus = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "6060"

    hps = utils.get_hparams()
    mp.spawn(
        run,
        nprocs=n_gpus,
        args=(
            n_gpus,
            hps,
        ),
    )


def run(rank, n_gpus, hps):
    global global_step
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=n_gpus, rank=rank
    )
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)

    vocoder_only = bool(getattr(hps.model, "vocoder_only", False) or getattr(hps.model, "vae2_mode", False))
    vae2_kl_weight = getattr(hps.model, "vae2_kl_weight", 1e-4)
    
    print(f"VAE2 bottleneck mode: latent_dim={getattr(hps.model, 'vae2_latent_dim', 128)}, "
          f"downsample={getattr(hps.model, 'vae2_downsample', 4)}x, kl_weight={vae2_kl_weight}")

    if (
        "use_mel_posterior_encoder" in hps.model.keys()
        and hps.model.use_mel_posterior_encoder == True
    ):
        print("Using mel posterior encoder for VITS2")
        posterior_channels = hps.data.n_mel_channels
        hps.data.use_mel_posterior_encoder = True
    else:
        print("Using lin posterior encoder for VITS1")
        posterior_channels = hps.data.filter_length // 2 + 1
        hps.data.use_mel_posterior_encoder = False

    if vocoder_only:
        print("Running in vocoder_only mode: text encoder, MAS/flow, and duration modules are disabled.")

    train_dataset = TextAudioLoader(hps.data.training_files, hps.data)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size,
        [32, 300, 400, 500, 600, 700, 800, 900, 1000],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )

    collate_fn = TextAudioCollate()
    train_loader = DataLoader(
        train_dataset,
        num_workers=8,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=True if n_gpus > 0 else False,
        prefetch_factor=2,
        worker_init_fn=worker_init_fn,
    )
    if rank == 0:
        eval_dataset = TextAudioLoader(hps.data.validation_files, hps.data)
        eval_loader = DataLoader(
            eval_dataset,
            num_workers=8,
            shuffle=False,
            batch_size=hps.train.batch_size,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
            persistent_workers=True,
            prefetch_factor=2,
            worker_init_fn=worker_init_fn,
        )

    # Vocoder-only mode: no text encoder, flows, or duration predictor
    net_g = SynthesizerTrn(
        len(symbols),
        posterior_channels,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model,
    ).cuda(rank)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)
    net_cqtd = MultiScaleSubbandCQTDiscriminator(hps).cuda(rank)
    
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(
        itertools.chain(net_d.parameters(), net_cqtd.parameters()),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )

    net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=False)
    net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=False)
    net_cqtd = DDP(net_cqtd, device_ids=[rank], find_unused_parameters=False)

    try:
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g
        )
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), [net_d, net_cqtd], optim_d
        )
        global_step = (epoch_str - 1) * len(train_loader)
    except:
        epoch_str = 1
        global_step = 0

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )

    scaler = GradScaler(enabled=hps.train.fp16_run)

    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d, net_cqtd],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                [train_loader, eval_loader],
                logger,
                [writer, writer_eval],
                vae2_kl_weight,
            )
        else:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d, net_cqtd],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                [train_loader, None],
                None,
                None,
                vae2_kl_weight,
            )
        scheduler_g.step()
        scheduler_d.step()


def train_and_evaluate(
    rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers,
    vae2_kl_weight=1e-4
):
    net_g, net_d, net_cqtd = nets
    optim_g, optim_d = optims
    scheduler_g, scheduler_d = schedulers
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()
    net_cqtd.train()

    if rank == 0:
        loader = tqdm.tqdm(train_loader, desc="Loading train data")
    else:
        loader = train_loader
    for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths) in enumerate(
        loader
    ):
        # Vocoder only uses spec and y (mel and audio)
        spec, spec_lengths = spec.cuda(rank, non_blocking=True), spec_lengths.cuda(
            rank, non_blocking=True
        )
        y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(
            rank, non_blocking=True
        )
        # x is not used in vocoder_only mode, but we still need dummy values for the forward call
        x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(
            rank, non_blocking=True
        )

        with autocast(enabled=hps.train.fp16_run):
            (
                y_hat,
                l_length,
                attn,
                ids_slice,
                x_mask,
                z_mask,
                (z, z_p, m_p, logs_p, m_q, logs_q),
                (hidden_x, logw, logw_),
                (mag, phase)
            ) = net_g(x, x_lengths, spec, spec_lengths)

            if (
                hps.model.use_mel_posterior_encoder
                or hps.data.use_mel_posterior_encoder
            ):
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
            y_mel = commons.slice_segments(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length
            )

            y = commons.slice_segments(
                y, ids_slice * hps.data.hop_length, hps.train.segment_size
            )  # slice
            
            # Match y_hat length to y to avoid discriminator feature map size mismatch
            if y_hat.size(-1) > y.size(-1):
                y_hat = y_hat[:, :, :y.size(-1)]
            elif y_hat.size(-1) < y.size(-1):
                y_hat = F.pad(y_hat, (0, y.size(-1) - y_hat.size(-1)))

            # Compute y_hat_mel AFTER matching lengths, and match to y_mel shape
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
            # Ensure mel shapes match (STFT boundary effects)
            min_mel_len = min(y_mel.size(2), y_hat_mel.size(2))
            y_mel = y_mel[:, :, :min_mel_len]
            y_hat_mel = y_hat_mel[:, :, :min_mel_len]
            
            # STFT operations need float32 precision to avoid NaN in fp16 training
            with autocast(enabled=False):
                reshaped_y = y.view(-1, y.size(-1)).float()
                reshaped_y_hat = y_hat.view(-1, y_hat.size(-1)).float()
                # Use generator's STFT params to match mag shape from decoder (iSTFTNet architecture)
                y_stft = torch.stft(reshaped_y, n_fft=hps.model.gen_istft_n_fft, hop_length=hps.model.gen_istft_hop_size, win_length=hps.model.gen_istft_n_fft, return_complex=True)
                y_hat_stft = torch.stft(reshaped_y_hat, n_fft=hps.model.gen_istft_n_fft, hop_length=hps.model.gen_istft_hop_size, win_length=hps.model.gen_istft_n_fft, return_complex=True)
                # Truncate to minimum time length to handle STFT boundary differences (shape: [B, freq, time])
                min_t = min(mag.size(2), y_stft.size(2))
                target_magnitude = torch.abs(y_stft[:, :, :min_t])
                mag = mag[:, :, :min_t].float()
                y_stft = y_stft[:, :, :min_t]
                y_hat_stft = y_hat_stft[:, :, :min_t]
            
            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            y_d_rs, y_d_gs, _, _ = net_cqtd(y, y_hat.detach())
            
            with autocast(enabled=False):
                loss_disc, _, _ = discriminator_loss(
                    y_d_hat_r, y_d_hat_g
                )
                loss_cqt_disc, _, _ = discriminator_loss(
                    y_d_rs, y_d_gs
                )
                loss_disc_all = loss_disc + loss_cqt_disc

        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with autocast(enabled=hps.train.fp16_run):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            _, y_cqtd_hat_g, cqt_fmap_r, cqt_fmap_g = net_cqtd(y, y_hat)
            with autocast(enabled=False):
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel

                loss_magnitude = torch.nn.functional.l1_loss(mag, target_magnitude)
                loss_phase = phase_loss(y_stft, y_hat_stft)
                loss_sd = (loss_magnitude + loss_phase) * hps.train.c_sd
                
                loss_fm = feature_loss(fmap_r, fmap_g) + feature_loss(cqt_fmap_r, cqt_fmap_g)
                loss_mpd_gen, losses_mpd_gen = generator_loss(y_d_hat_g)
                loss_cqtd_gen, losses_cqtd_gen = generator_loss(y_cqtd_hat_g)
                loss_gen = loss_mpd_gen + loss_cqtd_gen
                
                # VAE2 bottleneck mode - l_length contains KL loss
                loss_kl = l_length * vae2_kl_weight
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl

        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]["lr"]
                losses = [loss_disc_all, loss_gen, loss_fm, loss_mel, loss_sd, loss_kl]
                logger.info(
                    "Train Epoch: {} [{:.0f}%]".format(
                        epoch, 100.0 * batch_idx / len(train_loader)
                    )
                )
                logger.info([x.item() for x in losses] + [global_step, lr])

                scalar_dict = {
                    "loss/g/total": loss_gen_all,
                    "loss/d/total": loss_disc_all,
                    "learning_rate": lr,
                    "grad_norm_d": grad_norm_d,
                    "grad_norm_g": grad_norm_g,
                }
                scalar_dict.update(
                    {
                        "loss/g/fm": loss_fm,
                        "loss/g/mel": loss_mel,
                        "loss/g/sd": loss_sd,
                        "loss/g/magnitude": loss_magnitude,
                        "loss/g/phase": loss_phase,
                        "loss/g/kl": loss_kl,
                        "loss/g/kl_raw": l_length,
                    }
                )

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

            if global_step % hps.train.eval_interval == 0:
                evaluate(hps, net_g, eval_loader, writer_eval)
                utils.save_checkpoint(
                    net_g,
                    optim_g,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(hps.model_dir, "G_{}.pth".format(global_step)),
                )
                utils.save_checkpoint(
                    [net_d, net_cqtd],
                    optim_d,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(hps.model_dir, "D_{}.pth".format(global_step)),
                )
        global_step += 1

    if rank == 0:
        logger.info("====> Epoch: {}".format(epoch))


def evaluate(hps, generator, eval_loader, writer_eval):
    generator.eval()

    num_eval_samples = int(getattr(hps.train, "num_eval_samples", 15))
    num_log_samples = int(getattr(hps.train, "num_eval_log_samples", 3))
    if num_eval_samples <= 0 or num_log_samples <= 0:
        generator.train()
        return

    num_log_samples = min(num_log_samples, num_eval_samples)
    device = next(generator.parameters()).device

    image_dict = {}
    audio_dict = {}
    logged = 0

    with torch.no_grad():
        total_processed = 0
        for _, (_, _, spec, spec_lengths, y, y_lengths) in enumerate(eval_loader):
            spec = spec.to(device, non_blocking=True)
            spec_lengths = spec_lengths.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_lengths = y_lengths.to(device, non_blocking=True)

            batch_size = spec.size(0)
            for sample_idx in range(batch_size):
                if total_processed >= num_eval_samples or logged >= num_log_samples:
                    break

                spec_i = spec[sample_idx : sample_idx + 1]
                spec_len_i = spec_lengths[sample_idx : sample_idx + 1]
                y_i = y[sample_idx : sample_idx + 1]
                y_len_i = y_lengths[sample_idx : sample_idx + 1]

                y_hat_i, mask_i, _ = generator.module.vocoder_infer(
                    spec_i, spec_len_i, max_len=None
                )
                y_hat_len_i = (mask_i.sum([1, 2]).long() * hps.data.hop_length).item()

                if hps.model.use_mel_posterior_encoder or hps.data.use_mel_posterior_encoder:
                    mel_i = spec_i
                else:
                    mel_i = spec_to_mel_torch(
                        spec_i,
                        hps.data.filter_length,
                        hps.data.n_mel_channels,
                        hps.data.sampling_rate,
                        hps.data.mel_fmin,
                        hps.data.mel_fmax,
                    )

                mel_len_i = int(spec_len_i.item())
                mel_i = mel_i[:, :, :mel_len_i]
                y_i = y_i[:, :, : int(y_len_i.item())]
                y_hat_i = y_hat_i[:, :, :y_hat_len_i]

                y_hat_mel_i = mel_spectrogram_torch(
                    y_hat_i.squeeze(1).float(),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )

                image_dict[f"gen/mel_{logged}"] = utils.plot_spectrogram_to_numpy(
                    y_hat_mel_i[0].cpu().numpy()
                )
                audio_dict[f"gen/audio_{logged}"] = y_hat_i[0]

                if global_step == 0:
                    image_dict[f"gt/mel_{logged}"] = utils.plot_spectrogram_to_numpy(
                        mel_i[0].cpu().numpy()
                    )
                    audio_dict[f"gt/audio_{logged}"] = y_i[0]

                logged += 1
                total_processed += 1

            if total_processed >= num_eval_samples or logged >= num_log_samples:
                break

    if logged == 0:
        generator.train()
        return

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
