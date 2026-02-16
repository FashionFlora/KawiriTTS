import copy
import math

import numpy as np
import torch
from torch import nn
from torch.nn import AvgPool1d, Conv1d, Conv2d, ConvTranspose1d
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, spectral_norm, weight_norm
import torchaudio.transforms as T

import einops
import attentions
import commons
import modules
try:
    import monotonic_align
except ModuleNotFoundError:
    monotonic_align = None

from stft import stft
from stft import TorchSTFT
from conformer.conformer import Conformer
from commons import get_padding, init_weights
from generator_blocks import *
from einops import rearrange
from nnAudio import features

from typing import List, Tuple

LRELU_SLOPE = 0.1

AVAILABLE_FLOW_TYPES = [
    "pre_conv",
    "pre_conv2",
    "fft",
    "mono_layer_inter_residual",
    "mono_layer_post_residual",
]

AVAILABLE_DURATION_DISCRIMINATOR_TYPES = [
    "dur_disc_1",
    "dur_disc_2",
]

class StochasticDurationPredictor(nn.Module):
    def __init__(
        self,
        in_channels,
        filter_channels,
        kernel_size,
        p_dropout,
        n_flows=4,
        gin_channels=0,
    ):
        super().__init__()
        filter_channels = in_channels  # it needs to be removed from future version.
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.log_flow = modules.Log()
        self.flows = nn.ModuleList()
        self.flows.append(modules.ElementwiseAffine(2))
        for i in range(n_flows):
            self.flows.append(
                modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3)
            )
            self.flows.append(modules.Flip())

        self.post_pre = nn.Conv1d(1, filter_channels, 1)
        self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.post_convs = modules.DDSConv(
            filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout
        )
        self.post_flows = nn.ModuleList()
        self.post_flows.append(modules.ElementwiseAffine(2))
        for i in range(4):
            self.post_flows.append(
                modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3)
            )
            self.post_flows.append(modules.Flip())

        self.pre = nn.Conv1d(in_channels, filter_channels, 1)
        self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.convs = modules.DDSConv(
            filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout
        )
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, filter_channels, 1)

    def forward(self, x, x_mask, w=None, g=None, reverse=False, noise_scale=1.0):
        x = torch.detach(x)
        x = self.pre(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.convs(x, x_mask)
        x = self.proj(x) * x_mask

        if not reverse:
            flows = self.flows
            assert w is not None

            logdet_tot_q = 0
            h_w = self.post_pre(w)
            h_w = self.post_convs(h_w, x_mask)
            h_w = self.post_proj(h_w) * x_mask
            e_q = (
                torch.randn(w.size(0), 2, w.size(2)).to(device=x.device, dtype=x.dtype)
                * x_mask
            )
            z_q = e_q
            for flow in self.post_flows:
                z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
                logdet_tot_q += logdet_q
            z_u, z1 = torch.split(z_q, [1, 1], 1)
            u = torch.sigmoid(z_u) * x_mask
            z0 = (w - u) * x_mask
            logdet_tot_q += torch.sum(
                (F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1, 2]
            )
            logq = (
                torch.sum(-0.5 * (math.log(2 * math.pi) + (e_q**2)) * x_mask, [1, 2])
                - logdet_tot_q
            )

            logdet_tot = 0
            z0, logdet = self.log_flow(z0, x_mask)
            logdet_tot += logdet
            z = torch.cat([z0, z1], 1)
            for flow in flows:
                z, logdet = flow(z, x_mask, g=x, reverse=reverse)
                logdet_tot = logdet_tot + logdet
            nll = (
                torch.sum(0.5 * (math.log(2 * math.pi) + (z**2)) * x_mask, [1, 2])
                - logdet_tot
            )
            return nll + logq  # [b]
        else:
            flows = list(reversed(self.flows))
            flows = flows[:-2] + [flows[-1]]  # remove a useless vflow
            z = (
                torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype)
                * noise_scale
            )
            for flow in flows:
                z = flow(z, x_mask, g=x, reverse=reverse)
            z0, z1 = torch.split(z, [1, 1], 1)
            logw = z0
            return logw


class DurationPredictor(nn.Module):
    def __init__(
        self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0
    ):
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_1 = modules.LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_2 = modules.LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

    def forward(self, x, x_mask, g=None):
        x = torch.detach(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask


class DurationDiscriminatorV1(nn.Module):  # vits2
    # TODO : not using "spk conditioning" for now according to the paper.
    # Can be a better discriminator if we use it.
    def __init__(
        self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0
    ):
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        # self.norm_1 = modules.LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        # self.norm_2 = modules.LayerNorm(filter_channels)
        self.dur_proj = nn.Conv1d(1, filter_channels, 1)

        self.pre_out_conv_1 = nn.Conv1d(
            2 * filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.pre_out_norm_1 = modules.LayerNorm(filter_channels)
        self.pre_out_conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.pre_out_norm_2 = modules.LayerNorm(filter_channels)

        # if gin_channels != 0:
        #   self.cond = nn.Conv1d(gin_channels, in_channels, 1)

        self.output_layer = nn.Sequential(nn.Linear(filter_channels, 1), nn.Sigmoid())

    def forward_probability(self, x, x_mask, dur, g=None):
        dur = self.dur_proj(dur)
        x = torch.cat([x, dur], dim=1)
        x = self.pre_out_conv_1(x * x_mask)
        # x = torch.relu(x)
        # x = self.pre_out_norm_1(x)
        # x = self.drop(x)
        x = self.pre_out_conv_2(x * x_mask)
        # x = torch.relu(x)
        # x = self.pre_out_norm_2(x)
        # x = self.drop(x)
        x = x * x_mask
        x = x.transpose(1, 2)
        output_prob = self.output_layer(x)
        return output_prob

    def forward(self, x, x_mask, dur_r, dur_hat, g=None):
        x = torch.detach(x)
        # if g is not None:
        #   g = torch.detach(g)
        #   x = x + self.cond(g)
        x = self.conv_1(x * x_mask)
        # x = torch.relu(x)
        # x = self.norm_1(x)
        # x = self.drop(x)
        x = self.conv_2(x * x_mask)
        # x = torch.relu(x)
        # x = self.norm_2(x)
        # x = self.drop(x)

        output_probs = []
        for dur in [dur_r, dur_hat]:
            output_prob = self.forward_probability(x, x_mask, dur, g)
            output_probs.append(output_prob)

        return output_probs


class DurationDiscriminatorV2(nn.Module):  # vits2
    # TODO : not using "spk conditioning" for now according to the paper.
    # Can be a better discriminator if we use it.
    def __init__(
        self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0
    ):
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.conv_1 = nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_1 = modules.LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_2 = modules.LayerNorm(filter_channels)
        self.dur_proj = nn.Conv1d(1, filter_channels, 1)

        self.pre_out_conv_1 = nn.Conv1d(
            2 * filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.pre_out_norm_1 = modules.LayerNorm(filter_channels)
        self.pre_out_conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.pre_out_norm_2 = modules.LayerNorm(filter_channels)

        # if gin_channels != 0:
        #   self.cond = nn.Conv1d(gin_channels, in_channels, 1)

        self.output_layer = nn.Sequential(nn.Linear(filter_channels, 1), nn.Sigmoid())

    def forward_probability(self, x, x_mask, dur, g=None):
        dur = self.dur_proj(dur)
        x = torch.cat([x, dur], dim=1)
        x = self.pre_out_conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.pre_out_norm_1(x)
        x = self.pre_out_conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.pre_out_norm_2(x)
        x = x * x_mask
        x = x.transpose(1, 2)
        output_prob = self.output_layer(x)
        return output_prob

    def forward(self, x, x_mask, dur_r, dur_hat, g=None):
        x = torch.detach(x)
        # if g is not None:
        #   g = torch.detach(g)
        #   x = x + self.cond(g)
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)

        output_probs = []
        for dur in [dur_r, dur_hat]:
            output_prob = self.forward_probability(x, x_mask, dur, g)
            output_probs.append([output_prob])

        return output_probs


class TextEncoder(nn.Module):
    def __init__(
        self,
        n_vocab,
        out_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        gin_channels=0,
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels
        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

        self.encoder = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            gin_channels=self.gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        x = self.emb(x) * math.sqrt(self.hidden_channels)  # [b, t, h]
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )

        x = self.encoder(x * x_mask, x_mask, g=g)
        stats = self.proj(x) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask


class ResidualCouplingTransformersLayer2(nn.Module):  # vits2
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        p_dropout=0,
        gin_channels=0,
        mean_only=False,
    ):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.pre_transformer = attentions.Encoder(
            hidden_channels,
            hidden_channels,
            n_heads=2,
            n_layers=1,
            kernel_size=kernel_size,
            p_dropout=p_dropout,
            # window_size=None,
        )
        self.enc = modules.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            p_dropout=p_dropout,
            gin_channels=gin_channels,
        )

        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask
        h = h + self.pre_transformer(h * x_mask, x_mask)  # vits2 residual connection
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)
        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], 1)
            return x


class ResidualCouplingTransformersLayer(nn.Module):  # vits2
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        p_dropout=0,
        gin_channels=0,
        mean_only=False,
    ):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only
        # vits2
        self.pre_transformer = attentions.Encoder(
            self.half_channels,
            self.half_channels,
            n_heads=2,
            n_layers=2,
            kernel_size=3,
            p_dropout=0.1,
            window_size=None,
        )

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = modules.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            p_dropout=p_dropout,
            gin_channels=gin_channels,
        )
        # vits2
        self.post_transformer = attentions.Encoder(
            self.hidden_channels,
            self.hidden_channels,
            n_heads=2,
            n_layers=2,
            kernel_size=3,
            p_dropout=0.1,
            window_size=None,
        )

        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        x0_ = self.pre_transformer(x0 * x_mask, x_mask)  # vits2
        x0_ = x0_ + x0  # vits2 residual connection
        h = self.pre(x0_) * x_mask  # changed from x0 to x0_ to retain x0 for the flow
        h = self.enc(h, x_mask, g=g)

        # vits2 - (experimental;uncomment the following 2 line to use)
        # h_ = self.post_transformer(h, x_mask)
        # h = h + h_ #vits2 residual connection

        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)
        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], 1)
            return x


class FFTransformerCouplingLayer(nn.Module):  # vits2
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        n_layers,
        n_heads,
        p_dropout=0,
        filter_channels=768,
        mean_only=False,
        gin_channels=0,
    ):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = attentions.FFT(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            isflow=True,
            gin_channels=gin_channels,
        )
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask
        h_ = self.enc(h, x_mask, g=g)
        h = h_ + h
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], 1)
            return x


class MonoTransformerFlowLayer(nn.Module):  # vits2
    def __init__(
        self,
        channels,
        hidden_channels,
        mean_only=False,
        residual_connection=False,
        # according to VITS-2 paper fig 1B set residual_connection=True
    ):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.half_channels = channels // 2
        self.mean_only = mean_only
        self.residual_connection = residual_connection
        # vits2
        self.pre_transformer = attentions.Encoder(
            self.half_channels,
            self.half_channels,
            n_heads=2,
            n_layers=2,
            kernel_size=3,
            p_dropout=0.1,
            window_size=None,
        )

        self.post = nn.Conv1d(
            self.half_channels, self.half_channels * (2 - mean_only), 1
        )
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        if self.residual_connection:
            if not reverse:
                x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
                x0_ = self.pre_transformer(x0, x_mask)  # vits2
                stats = self.post(x0_) * x_mask
                if not self.mean_only:
                    m, logs = torch.split(stats, [self.half_channels] * 2, 1)
                else:
                    m = stats
                    logs = torch.zeros_like(m)
                x1 = m + x1 * torch.exp(logs) * x_mask
                x_ = torch.cat([x0, x1], 1)
                x = x + x_
                logdet = torch.sum(torch.log(torch.exp(logs) + 1), [1, 2])
                logdet = logdet + torch.log(torch.tensor(2)) * (
                    x0.shape[1] * x0.shape[2]
                )
                return x, logdet
            else:
                x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
                x0 = x0 / 2
                x0_ = x0 * x_mask
                x0_ = self.pre_transformer(x0, x_mask)  # vits2
                stats = self.post(x0_) * x_mask
                if not self.mean_only:
                    m, logs = torch.split(stats, [self.half_channels] * 2, 1)
                else:
                    m = stats
                    logs = torch.zeros_like(m)
                x1_ = ((x1 - m) / (1 + torch.exp(-logs))) * x_mask
                x = torch.cat([x0, x1_], 1)
                return x
        else:
            x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
            x0_ = self.pre_transformer(x0 * x_mask, x_mask)  # vits2
            h = x0_ + x0  # vits2
            stats = self.post(h) * x_mask
            if not self.mean_only:
                m, logs = torch.split(stats, [self.half_channels] * 2, 1)
            else:
                m = stats
                logs = torch.zeros_like(m)
            if not reverse:
                x1 = m + x1 * torch.exp(logs) * x_mask
                x = torch.cat([x0, x1], 1)
                logdet = torch.sum(logs, [1, 2])
                return x, logdet
            else:
                x1 = (x1 - m) * torch.exp(-logs) * x_mask
                x = torch.cat([x0, x1], 1)
                return x


class ResidualCouplingTransformersBlock(nn.Module):  # vits2
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        n_flows=4,
        gin_channels=0,
        use_transformer_flows=False,
        transformer_flow_type="pre_conv",
    ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        if use_transformer_flows:
            if transformer_flow_type == "pre_conv":
                for i in range(n_flows):
                    self.flows.append(
                        ResidualCouplingTransformersLayer(
                            channels,
                            hidden_channels,
                            kernel_size,
                            dilation_rate,
                            n_layers,
                            gin_channels=gin_channels,
                            mean_only=True,
                        )
                    )
                    self.flows.append(modules.Flip())
            elif transformer_flow_type == "pre_conv2":
                for i in range(n_flows):
                    self.flows.append(
                        ResidualCouplingTransformersLayer2(
                            channels,
                            hidden_channels,
                            kernel_size,
                            dilation_rate,
                            n_layers,
                            gin_channels=gin_channels,
                            mean_only=True,
                        )
                    )
                    self.flows.append(modules.Flip())
            elif transformer_flow_type == "fft":
                for i in range(n_flows):
                    self.flows.append(
                        FFTransformerCouplingLayer(
                            channels,
                            hidden_channels,
                            kernel_size,
                            dilation_rate,
                            n_layers,
                            gin_channels=gin_channels,
                            mean_only=True,
                        )
                    )
                    self.flows.append(modules.Flip())
            elif transformer_flow_type == "mono_layer_inter_residual":
                for i in range(n_flows):
                    self.flows.append(
                        modules.ResidualCouplingLayer(
                            channels,
                            hidden_channels,
                            kernel_size,
                            dilation_rate,
                            n_layers,
                            gin_channels=gin_channels,
                            mean_only=True,
                        )
                    )
                    self.flows.append(modules.Flip())
                    self.flows.append(
                        MonoTransformerFlowLayer(
                            channels, hidden_channels, mean_only=True
                        )
                    )
            elif transformer_flow_type == "mono_layer_post_residual":
                for i in range(n_flows):
                    self.flows.append(
                        modules.ResidualCouplingLayer(
                            channels,
                            hidden_channels,
                            kernel_size,
                            dilation_rate,
                            n_layers,
                            gin_channels=gin_channels,
                            mean_only=True,
                        )
                    )
                    self.flows.append(modules.Flip())
                    self.flows.append(
                        MonoTransformerFlowLayer(
                            channels,
                            hidden_channels,
                            mean_only=True,
                            residual_connection=True,
                        )
                    )
        else:
            for i in range(n_flows):
                self.flows.append(
                    modules.ResidualCouplingLayer(
                        channels,
                        hidden_channels,
                        kernel_size,
                        dilation_rate,
                        n_layers,
                        gin_channels=gin_channels,
                        mean_only=True,
                    )
                )
                self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        n_flows=4,
        gin_channels=0,
    ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                modules.ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class VAE2Encoder(nn.Module):
    """
    VAE2 Encoder: Encodes mel spectrogram to latent space with configurable downsampling.
    Uses KL divergence loss for regularization.
    """
    def __init__(
        self,
        in_channels,
        latent_dim=128,
        hidden_channels=256,
        downsample_factor=4,
        gin_channels=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels
        self.downsample_factor = downsample_factor
        self.gin_channels = gin_channels

        if self.downsample_factor < 1 or (
            self.downsample_factor & (self.downsample_factor - 1)
        ) != 0:
            raise ValueError(
                f"downsample_factor must be a positive power of 2, got {self.downsample_factor}"
            )
        self.num_downsample_layers = int(math.log2(self.downsample_factor))

        # Initial projection
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        
        # Encoding layers with downsampling (100Hz -> 25Hz, 4x down)
        # Using strided convolutions for downsampling
        self.enc_layers = nn.ModuleList()
        
        # Downsample in repeated 2x stages
        for _ in range(self.num_downsample_layers):
            self.enc_layers.append(nn.Sequential(
                weight_norm(nn.Conv1d(hidden_channels, hidden_channels, 4, stride=2, padding=1)),
                nn.LeakyReLU(0.1),
            ))
        
        # Additional processing at 25Hz
        self.enc_proc = nn.Sequential(
            weight_norm(nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1)),
            nn.LeakyReLU(0.1),
            weight_norm(nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1)),
            nn.LeakyReLU(0.1),
        )
        
        # Output mean and log_var for KL divergence
        self.proj = nn.Conv1d(hidden_channels, latent_dim * 2, 1)
        
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, hidden_channels, 1)
        else:
            self.cond = None

    def forward(self, x, x_lengths, g=None):
        """
        Args:
            x: [B, mel_channels, T] mel spectrogram at 100Hz
            x_lengths: [B] lengths
            g: [B, gin_channels, 1] speaker embedding
        Returns:
            z: [B, latent_dim, T//4] sampled latent at 25Hz
            mean: [B, latent_dim, T//4] mean
            log_var: [B, latent_dim, T//4] log variance
            z_mask: [B, 1, T//4] mask at 25Hz
        """
        # Create mask for input (100Hz)
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        
        # Pre-process
        x = self.pre(x) * x_mask
        
        if g is not None and self.cond is not None:
            x = x + self.cond(g)
        
        # Downsample through encoding layers
        for enc_layer in self.enc_layers:
            x = enc_layer(x)
        
        # Process at 25Hz
        x = self.enc_proc(x)
        
        # Create mask for latent (25Hz)
        z_lengths = x_lengths // self.downsample_factor
        z_mask = torch.unsqueeze(commons.sequence_mask(z_lengths, x.size(2)), 1).to(x.dtype)
        
        # Project to mean and log_var
        stats = self.proj(x) * z_mask
        mean, log_var = torch.split(stats, self.latent_dim, dim=1)
        
        # Reparameterization trick
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mean + eps * std
        z = z * z_mask
        
        return z, mean, log_var, z_mask, z_lengths


class VAE2EncoderV2(nn.Module):
    """
    VAE2 Encoder V2: Enhanced encoder with larger receptive field and capacity.
    
    Simplified version using attentions.Encoder (standard transformer) instead of Conformer
    for better gradient stability. Uses fewer WN blocks with residual scaling.
    
    Downsamples mel spectrogram with configurable power-of-two downsampling.
    """
    def __init__(
        self,
        in_channels,
        latent_dim=128,
        hidden_channels=256,
        downsample_factor=4,
        gin_channels=0,
        # WN block config
        wn_kernel_size=5,
        wn_dilation_rate=1,
        # Number of WN blocks at each rate (reduced for stability)
        n_blocks_100hz=2,  # 2 blocks at 100Hz
        n_blocks_50hz=2,   # 2 blocks at 50Hz  
        n_blocks_25hz=3,   # 3 blocks at 25Hz
        # Transformer config at 25Hz
        transformer_n_layers=2,
        transformer_n_heads=4,
        transformer_filter_channels=512,
        p_dropout=0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels
        self.downsample_factor = downsample_factor
        self.gin_channels = gin_channels

        if self.downsample_factor < 1 or (
            self.downsample_factor & (self.downsample_factor - 1)
        ) != 0:
            raise ValueError(
                f"downsample_factor must be a positive power of 2, got {self.downsample_factor}"
            )
        self.num_downsample_layers = int(math.log2(self.downsample_factor))

        # Initial projection from mel channels to hidden
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        
        # ============ 100Hz Stage ============
        # WN blocks at 100Hz for local feature extraction
        self.wn_100hz = nn.ModuleList()
        for i in range(n_blocks_100hz):
            self.wn_100hz.append(
                modules.WN(
                    hidden_channels=hidden_channels,
                    kernel_size=wn_kernel_size,
                    dilation_rate=wn_dilation_rate,
                    n_layers=3,  # reduced from 4
                    gin_channels=gin_channels if i == 0 else 0,
                    p_dropout=p_dropout,
                )
            )
        self.norm_100hz = modules.LayerNorm(hidden_channels)
        
        # Configurable 2x downsample stages
        self.downsample_layers = nn.ModuleList()
        for _ in range(self.num_downsample_layers):
            self.downsample_layers.append(
                nn.Sequential(
                    weight_norm(nn.Conv1d(hidden_channels, hidden_channels, 4, stride=2, padding=1)),
                    nn.LeakyReLU(0.1),
                )
            )
        
        # ============ 50Hz Stage ============
        # Intermediate stage blocks (applied after each downsample stage except final latent stage)
        self.wn_mid = nn.ModuleList()
        self.norm_mid = nn.ModuleList()
        for _ in range(max(self.num_downsample_layers - 1, 0)):
            stage_blocks = nn.ModuleList()
            for _ in range(n_blocks_50hz):
                stage_blocks.append(
                    modules.WN(
                        hidden_channels=hidden_channels,
                        kernel_size=wn_kernel_size,
                        dilation_rate=wn_dilation_rate,
                        n_layers=3,
                        gin_channels=0,
                        p_dropout=p_dropout,
                    )
                )
            self.wn_mid.append(stage_blocks)
            self.norm_mid.append(modules.LayerNorm(hidden_channels))
        
        # ============ Latent Stage ============
        # WN blocks at latent rate for deeper local processing
        self.wn_latent = nn.ModuleList()
        for _ in range(n_blocks_25hz):
            self.wn_latent.append(
                modules.WN(
                    hidden_channels=hidden_channels,
                    kernel_size=wn_kernel_size,
                    dilation_rate=wn_dilation_rate,
                    n_layers=3,
                    gin_channels=0,
                    p_dropout=p_dropout,
                )
            )
        self.norm_latent = modules.LayerNorm(hidden_channels)
        
        # Simple Transformer at 25Hz (replaces Conformer for stability)
        # Uses the existing attentions.Encoder which is well-tested
        self.transformer = attentions.Encoder(
            hidden_channels=hidden_channels,
            filter_channels=transformer_filter_channels,
            n_heads=transformer_n_heads,
            n_layers=transformer_n_layers,
            kernel_size=3,
            p_dropout=p_dropout,
            window_size=4,  # local attention window
        )
        
        # Post-transformer processing
        self.post = nn.Sequential(
            weight_norm(nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1)),
            nn.LeakyReLU(0.1),
        )
        
        # Output mean and log_var for KL divergence
        self.proj = nn.Conv1d(hidden_channels, latent_dim * 2, 1)
        # Initialize proj to small values for stable KL
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()
        
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, hidden_channels, 1)
        else:
            self.cond = None

    def forward(self, x, x_lengths, g=None):
        """
        Args:
            x: [B, mel_channels, T] mel spectrogram at 100Hz
            x_lengths: [B] lengths
            g: [B, gin_channels, 1] speaker embedding
        Returns:
            z: [B, latent_dim, T//downsample_factor] sampled latent
            mean: [B, latent_dim, T//downsample_factor] mean
            log_var: [B, latent_dim, T//downsample_factor] log variance
            z_mask: [B, 1, T//downsample_factor] mask
            z_lengths: [B] latent lengths
        """
        # Create mask for input (100Hz)
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        
        # Pre-process
        x = self.pre(x) * x_mask
        
        # Add speaker conditioning
        if g is not None and self.cond is not None:
            x = x + self.cond(g)
        
        # ============ 100Hz Stage ============
        for i, wn_block in enumerate(self.wn_100hz):
            if i == 0 and self.gin_channels != 0:
                x = x + 0.5 * wn_block(x, x_mask, g=g)  # residual scaling
            else:
                x = x + 0.5 * wn_block(x, x_mask)
        x = self.norm_100hz(x)
        
        # Downsample stages
        current_mask = x_mask
        for stage_idx, downsample_layer in enumerate(self.downsample_layers):
            x = downsample_layer(x)
            current_mask = current_mask[:, :, ::2]
            if current_mask.size(2) > x.size(2):
                current_mask = current_mask[:, :, :x.size(2)]
            elif current_mask.size(2) < x.size(2):
                current_mask = F.pad(current_mask, (0, x.size(2) - current_mask.size(2)))
            x = x * current_mask

            if stage_idx < len(self.wn_mid):
                for wn_block in self.wn_mid[stage_idx]:
                    x = x + 0.5 * wn_block(x, current_mask)
                x = self.norm_mid[stage_idx](x)

        # Create mask for latent stage
        z_lengths = x_lengths // self.downsample_factor
        z_mask = torch.unsqueeze(commons.sequence_mask(z_lengths, x.size(2)), 1).to(x.dtype)
        x = x * z_mask
        
        # ============ Latent Stage ============
        for wn_block in self.wn_latent:
            x = x + 0.5 * wn_block(x, z_mask)
        x = self.norm_latent(x)
        
        # Transformer at 25Hz (simple attention, very stable)
        x = self.transformer(x, z_mask)
        x = x * z_mask
        
        # Post-transformer processing
        x = self.post(x) * z_mask
        
        # Project to mean and log_var
        stats = self.proj(x) * z_mask
        mean, log_var = torch.split(stats, self.latent_dim, dim=1)
        
        # Clamp log_var for stability
        log_var = torch.clamp(log_var, min=-10.0, max=2.0)
        
        # Reparameterization trick
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mean + eps * std
        z = z * z_mask
        
        return z, mean, log_var, z_mask, z_lengths

    def remove_weight_norm(self):
        """Remove weight normalization for inference."""
        for wn_block in self.wn_100hz:
            wn_block.remove_weight_norm()
        for stage_blocks in self.wn_mid:
            for wn_block in stage_blocks:
                wn_block.remove_weight_norm()
        for wn_block in self.wn_latent:
            wn_block.remove_weight_norm()
        for downsample_layer in self.downsample_layers:
            torch.nn.utils.remove_weight_norm(downsample_layer[0])
        torch.nn.utils.remove_weight_norm(self.post[0])


class VAE2Upsampler(nn.Module):
    """
    Upsampler: Upsamples latent to mel frame rate using configurable power-of-two upsampling.
    This is placed between the VAE2 encoder and the vocoder decoder.
    """
    def __init__(
        self,
        latent_dim=128,
        out_channels=192,
        hidden_channels=256,
        upsample_factor=4,
        gin_channels=0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.upsample_factor = upsample_factor
        self.gin_channels = gin_channels

        if self.upsample_factor < 1 or (
            self.upsample_factor & (self.upsample_factor - 1)
        ) != 0:
            raise ValueError(
                f"upsample_factor must be a positive power of 2, got {self.upsample_factor}"
            )
        self.num_upsample_layers = int(math.log2(self.upsample_factor))
        
        # Process latent before upsampling
        self.pre = nn.Conv1d(latent_dim, hidden_channels, 1)
        
        # Upsample in repeated 2x stages
        self.up_layers = nn.ModuleList()
        for _ in range(self.num_upsample_layers):
            self.up_layers.append(nn.Sequential(
                weight_norm(nn.ConvTranspose1d(hidden_channels, hidden_channels, 4, stride=2, padding=1)),
                nn.LeakyReLU(0.1),
            ))
        
        # Additional processing at 100Hz
        self.post_proc = nn.Sequential(
            weight_norm(nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1)),
            nn.LeakyReLU(0.1),
            weight_norm(nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1)),
            nn.LeakyReLU(0.1),
        )
        
        # Project to vocoder input dimension (inter_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, hidden_channels, 1)
        else:
            self.cond = None

    def forward(self, z, z_mask, target_length=None, g=None):
        """
        Args:
            z: [B, latent_dim, T_z] latent features
            z_mask: [B, 1, T_z] latent mask
            target_length: int, target mel length (optional, for matching exact mel length)
            g: [B, gin_channels, 1] speaker embedding
        Returns:
            x: [B, out_channels, T] upsampled features
            x_mask: [B, 1, T] output mask
        """
        x = self.pre(z)
        
        if g is not None and self.cond is not None:
            x = x + self.cond(g)
        
        # Upsample through layers
        for up_layer in self.up_layers:
            x = up_layer(x)
        
        # Process at 100Hz
        x = self.post_proc(x)
        
        # Match target length if specified
        if target_length is not None:
            if x.size(2) > target_length:
                x = x[:, :, :target_length]
            elif x.size(2) < target_length:
                x = F.pad(x, (0, target_length - x.size(2)))
        
        # Create output mask
        out_length = x.size(2)
        x_mask = torch.ones(x.size(0), 1, out_length, device=x.device, dtype=x.dtype)
        
        # Project to output dimension
        x = self.proj(x) * x_mask
        
        return x, x_mask


def kl_divergence_loss(mean, log_var, mask):
    """
    Compute KL divergence loss for VAE.
    KL(q(z|x) || p(z)) where p(z) = N(0, I)
    
    Args:
        mean: [B, D, T]
        log_var: [B, D, T]
        mask: [B, 1, T]
    Returns:
        kl_loss: scalar, mean KL divergence per sample
    """
    # KL = -0.5 * sum(1 + log_var - mean^2 - exp(log_var))
    kl = -0.5 * (1 + log_var - mean.pow(2) - log_var.exp())
    kl = kl * mask
    # Sum over dimensions and time, mean over batch
    kl_loss = kl.sum(dim=[1, 2]).mean() / mask.sum(dim=[1, 2]).mean()
    return kl_loss


class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


def pixel_shuffle_1d(x: torch.Tensor, r: int) -> torch.Tensor:
    """1D pixel shuffle for upsampling."""
    # x: [B, C*r, L]
    B, Cr, L = x.size()
    C = Cr // r
    
    # Split channels into C groups of r
    x = x.view(B, C, r, L)
    
    # Permute to interleave: [B, C, L, r]
    x = x.permute(0, 1, 3, 2)
    
    # Flatten: [B, C, L*r]
    return x.reshape(B, C, L * r)


class UpsamplePixelShuffle1D(nn.Module):
    """Upsampling using 1D pixel shuffle (sub-pixel convolution)."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, r: int):
        super().__init__()
        self.r = r
        pad_l, pad_r = (kernel_size - 1) // 2, kernel_size // 2
        self.pad = nn.ReflectionPad1d((pad_l, pad_r))
        
        # We need out_ch * r channels to shuffle them into out_ch with r times spatial length
        self.conv = weight_norm(
            nn.Conv1d(in_ch, out_ch * r, kernel_size, padding=0)
        )
        
        # Apply ICNR initialization
        self._init_weights(in_ch, out_ch, r, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pad(x)
        x = self.conv(x)
        return pixel_shuffle_1d(x, self.r)

    def _init_weights(self, in_ch, out_ch, r, kernel_size):
        """
        ICNR Initialization:
        Initializes the convolution weights so that the PixelShuffle 
        starts as a smooth Nearest Neighbor upsample.
        """
        weight = self.conv.weight.data
        
        # Create a smaller kernel of shape [out_ch, in_ch, kernel_size]
        kernel = torch.zeros(out_ch, in_ch, kernel_size)
        nn.init.kaiming_normal_(kernel)
        
        # Expand by repeating 'r' times so channel groups share same weights initially
        weight.copy_(kernel.repeat(r, 1, 1))
        
        if self.conv.bias is not None:
            self.conv.bias.data.fill_(0)


class Generator(torch.nn.Module):
    def __init__(
        self,
        model_name,
        initial_channel,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gen_istft_n_fft,
        gen_istft_hop_size,
        gin_channels=0,
    ):
        super(Generator, self).__init__()
        self.model_name = model_name
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.gen_istft_n_fft = gen_istft_n_fft
        self.gen_istft_hop_size = gen_istft_hop_size
        self.conv_pre = Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, padding=3
        )
        resblock = modules.ResBlock1 if resblock == "1" else modules.ResBlock2
        
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            in_ch = upsample_initial_channel // (2**i)
            out_ch = upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(UpsamplePixelShuffle1D(in_ch, out_ch, kernel_size=k, r=u))

        self.alphas = nn.ParameterList()
        self.alphas.append(nn.Parameter(torch.ones(1, upsample_initial_channel, 1)))
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            self.alphas.append(nn.Parameter(torch.ones(1, ch, 1)))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock(ch, k, d))

        self.conformers = nn.ModuleList()
        self.post_n_fft = self.gen_istft_n_fft
        self.conv_post = weight_norm(Conv1d(128, self.post_n_fft + 2, 7, 1, padding=3))
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** i)
            self.conformers.append(Conformer(dim=ch, depth=2, dim_head=32, heads=8, ff_mult=4, conv_expansion_factor = 2, conv_kernel_size=31, attn_dropout=0.1, ff_dropout=0.1, conv_dropout=0.1))
        
        # UpsamplePixelShuffle1D uses ICNR init internally, no need for init_weights
        self.conv_post.apply(init_weights)
        self.reflection_pad = torch.nn.ReflectionPad1d((1, 0))
        self.stft = TorchSTFT("cuda", filter_length=self.gen_istft_n_fft, hop_length=self.gen_istft_hop_size, win_length=self.gen_istft_n_fft)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        # x: [b,d,t]
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = x + (1 / self.alphas[i]) * (torch.sin(self.alphas[i] * x) ** 2)
            x = einops.rearrange(x, 'b f t -> b t f')
            x = self.conformers[i](x)
            x = einops.rearrange(x, 'b t f -> b f t')

            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
                
        x = x + (1 / self.alphas[i+1]) * (torch.sin(self.alphas[i+1] * x) ** 2)
        x = self.reflection_pad(x)
        x = self.conv_post(x)
        
        spec = torch.exp(x[:,:self.post_n_fft // 2 + 1, :])
        phase = torch.sin(x[:, self.post_n_fft // 2 + 1:, :])
        out = self.stft.inverse(spec, phase).to(x.device)
        
        return out, spec, phase

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for l in self.ups:
            # UpsamplePixelShuffle1D has internal conv with weight_norm
            if hasattr(l, 'conv'):
                remove_weight_norm(l.conv)
            else:
                remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(
                    Conv2d(
                        1,
                        32,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        32,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        128,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        512,
                        1024,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        1024,
                        1024,
                        (kernel_size, 1),
                        1,
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
            ]
        )
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2, 3, 5, 7, 11]

        discs = [
            DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods
        ]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class SynthesizerTrn(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(
        self,
        n_vocab,
        spec_channels,
        segment_size,
        model_name,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gen_istft_n_fft=0,
        gen_istft_hop_size=0,
        n_speakers=0,
        gin_channels=0,
        use_sdp=True,
        **kwargs,
    ):
        super().__init__()
        self.model_name = model_name
        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gen_istft_n_fft = gen_istft_n_fft
        self.gen_istft_hop_size = gen_istft_hop_size
        self.gin_channels = gin_channels
        self.use_spk_conditioned_encoder = kwargs.get(
            "use_spk_conditioned_encoder", False
        )
        self.use_transformer_flows = kwargs.get("use_transformer_flows", False)
        self.vae2_mode = kwargs.get("vae2_mode", False)
        self.vocoder_only = kwargs.get("vocoder_only", False) or self.vae2_mode
        self.transformer_flow_type = kwargs.get(
            "transformer_flow_type", "mono_layer_post_residual"
        )
        if self.use_transformer_flows:
            assert (
                self.transformer_flow_type in AVAILABLE_FLOW_TYPES
            ), f"transformer_flow_type must be one of {AVAILABLE_FLOW_TYPES}"
        self.use_sdp = use_sdp
        # self.use_duration_discriminator = kwargs.get("use_duration_discriminator", False)
        self.use_noise_scaled_mas = kwargs.get("use_noise_scaled_mas", False)
        self.mas_noise_scale_initial = kwargs.get("mas_noise_scale_initial", 0.01)
        self.noise_scale_delta = kwargs.get("noise_scale_delta", 2e-6)

        self.current_mas_noise_scale = self.mas_noise_scale_initial
        if self.use_spk_conditioned_encoder and gin_channels > 0:
            self.enc_gin_channels = gin_channels
        else:
            self.enc_gin_channels = 0
        if not self.vocoder_only:
            self.enc_p = TextEncoder(
                n_vocab,
                inter_channels,
                hidden_channels,
                filter_channels,
                n_heads,
                n_layers,
                kernel_size,
                p_dropout,
                gin_channels=self.enc_gin_channels,
            )
        else:
            self.enc_p = None

        self.dec = Generator(
            model_name,
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gen_istft_n_fft,
            gen_istft_hop_size,
            gin_channels=gin_channels,
        )
        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels=gin_channels,
        )
        if not self.vocoder_only:
            self.flow = ResidualCouplingTransformersBlock(
                inter_channels,
                hidden_channels,
                5,
                1,
                4,
                gin_channels=gin_channels,
                use_transformer_flows=self.use_transformer_flows,
                transformer_flow_type=self.transformer_flow_type,
            )
        else:
            self.flow = None

        if self.vocoder_only:
            self.dp = None
        elif use_sdp:
            self.dp = StochasticDurationPredictor(
                hidden_channels, 192, 3, 0.5, 4, gin_channels=gin_channels
            )
        else:
            self.dp = DurationPredictor(
                hidden_channels, 256, 3, 0.5, gin_channels=gin_channels
            )

        if n_speakers > 1:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)
        
        # VAE2 bottleneck components
        self.use_vae2_bottleneck = kwargs.get("use_vae2_bottleneck", False)
        self.vae2_latent_dim = kwargs.get("vae2_latent_dim", 128)
        self.vae2_hidden_channels = kwargs.get("vae2_hidden_channels", 256)
        self.vae2_downsample = kwargs.get("vae2_downsample", 4)
        self.vae2_kl_weight = kwargs.get("vae2_kl_weight", 1e-4)
        
        if self.use_vae2_bottleneck:
            # ========== OLD VAE2Encoder (simple, ~260ms receptive field) ==========
            # self.vae2_encoder = VAE2Encoder(
            #     in_channels=spec_channels,
            #     latent_dim=self.vae2_latent_dim,
            #     hidden_channels=self.vae2_hidden_channels,
            #     downsample_factor=self.vae2_downsample,
            #     gin_channels=gin_channels,
            # )
            # ======================================================================
            
            # VAE2 encoder V2: mel -> latent
            # Uses WN blocks at each rate + Transformer at 25Hz for larger receptive field
            self.vae2_encoder = VAE2EncoderV2(
                in_channels=spec_channels,
                latent_dim=self.vae2_latent_dim,
                hidden_channels=self.vae2_hidden_channels,
                downsample_factor=self.vae2_downsample,
                gin_channels=gin_channels,
                # WN block config
                wn_kernel_size=kwargs.get("vae2_wn_kernel_size", 5),
                wn_dilation_rate=kwargs.get("vae2_wn_dilation_rate", 1),
                # Number of WN blocks at each rate
                n_blocks_100hz=kwargs.get("vae2_n_blocks_100hz", 2),
                n_blocks_50hz=kwargs.get("vae2_n_blocks_50hz", 2),
                n_blocks_25hz=kwargs.get("vae2_n_blocks_25hz", 3),
                # Transformer config at 25Hz
                transformer_n_layers=kwargs.get("vae2_transformer_n_layers", 2),
                transformer_n_heads=kwargs.get("vae2_transformer_n_heads", 4),
                transformer_filter_channels=kwargs.get("vae2_transformer_filter_channels", 512),
                p_dropout=kwargs.get("vae2_p_dropout", 0.1),
            )
            # VAE2 upsampler: latent -> vocoder input
            self.vae2_upsampler = VAE2Upsampler(
                latent_dim=self.vae2_latent_dim,
                out_channels=inter_channels,
                hidden_channels=self.vae2_hidden_channels,
                upsample_factor=self.vae2_downsample,
                gin_channels=gin_channels,
            )
            # Don't use PosteriorEncoder in VAE2 mode
            self.enc_q = None

    def forward(self, x, x_lengths, y, y_lengths, sid=None):
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            g = None

        # VAE2 bottleneck mode: mel -> encoder (downsample) -> latent -> upsampler -> vocoder
        if self.use_vae2_bottleneck:
            # Encode mel to latent space
            z_latent, mean, log_var, z_mask, z_lengths = self.vae2_encoder(y, y_lengths, g=g)
            
            # Upsample latent back to mel rate
            z_upsampled, y_mask = self.vae2_upsampler(z_latent, z_mask, target_length=y.size(2), g=g)
            
            # Random slice for vocoder training
            z_slice, ids_slice = commons.rand_slice_segments(
                z_upsampled, y_lengths, self.segment_size
            )
            o, mag, phase = self.dec(z_slice, g=g)
            
            # Compute KL loss
            kl_loss = kl_divergence_loss(mean, log_var, z_mask)
            
            # Dummy outputs for compatibility
            l_length = kl_loss  # We'll use l_length to pass KL loss
            attn = torch.zeros(
                y.size(0), 1, y.size(2), 1, device=y.device, dtype=y.dtype
            )
            x_mask = torch.ones(y.size(0), 1, 1, device=y.device, dtype=y.dtype)
            dummy_log = torch.zeros(y.size(0), 1, 1, device=y.device, dtype=y.dtype)
            
            return (
                o,
                l_length,  # KL loss
                attn,
                ids_slice,
                x_mask,
                y_mask,
                (z_latent, z_upsampled, mean, log_var, z_latent, z_latent),  # latent info
                (x_mask, dummy_log, dummy_log),
                (mag, phase),
            )

        if self.vocoder_only:
            z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
            z_slice, ids_slice = commons.rand_slice_segments(
                z, y_lengths, self.segment_size
            )
            o, mag, phase = self.dec(z_slice, g=g)

            zero_like_z = torch.zeros_like(z)
            l_length = torch.zeros(y.size(0), device=y.device, dtype=y.dtype)
            attn = torch.zeros(
                y.size(0),
                1,
                y.size(2),
                1,
                device=y.device,
                dtype=y.dtype,
            )
            x_mask = torch.ones(
                y.size(0),
                1,
                1,
                device=y.device,
                dtype=y.dtype,
            )
            dummy_log = torch.zeros(
                y.size(0),
                1,
                1,
                device=y.device,
                dtype=y.dtype,
            )
            return (
                o,
                l_length,
                attn,
                ids_slice,
                x_mask,
                y_mask,
                (z, z, zero_like_z, zero_like_z, m_q, logs_q),
                (x_mask, dummy_log, dummy_log),
                (mag, phase),
            )

        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths, g=g)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)

        if monotonic_align is None:
            raise RuntimeError(
                "monotonic_align is not available. Install/build it for text-aligned training, "
                "or enable vocoder_only/vae2_mode."
            )

        with torch.no_grad():
            # negative cross-entropy
            s_p_sq_r = torch.exp(-2 * logs_p)  # [b, d, t]
            neg_cent1 = torch.sum(
                -0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True
            )  # [b, 1, t_s]
            neg_cent2 = torch.matmul(
                -0.5 * (z_p**2).transpose(1, 2), s_p_sq_r
            )  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent3 = torch.matmul(
                z_p.transpose(1, 2), (m_p * s_p_sq_r)
            )  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent4 = torch.sum(
                -0.5 * (m_p**2) * s_p_sq_r, [1], keepdim=True
            )  # [b, 1, t_s]
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4

            if self.use_noise_scaled_mas:
                epsilon = (
                    torch.std(neg_cent)
                    * torch.randn_like(neg_cent)
                    * self.current_mas_noise_scale
                )
                neg_cent = neg_cent + epsilon

            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            attn = (
                monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1))
                .unsqueeze(1)
                .detach()
            )

        w = attn.sum(2)
        if self.use_sdp:
            l_length = self.dp(x, x_mask, w, g=g)
            l_length = l_length / torch.sum(x_mask)
            logw = self.dp(x, x_mask, g=g, reverse=True, noise_scale=1.0)
            logw_ = torch.log(w + 1e-6) * x_mask
        else:
            logw_ = torch.log(w + 1e-6) * x_mask
            logw = self.dp(x, x_mask, g=g)
            l_length = torch.sum((logw - logw_) ** 2, [1, 2]) / torch.sum(
                x_mask
            )  # for averaging

        # expand prior
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        z_slice, ids_slice = commons.rand_slice_segments(
            z, y_lengths, self.segment_size
        )
        o, mag, phase = self.dec(z_slice, g=g)
        return (
            o,
            l_length,
            attn,
            ids_slice,
            x_mask,
            y_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q),
            (x, logw, logw_),
            (mag, phase)
        )

    def infer(
        self,
        x,
        x_lengths,
        sid=None,
        noise_scale=1,
        length_scale=1,
        noise_scale_w=1.0,
        max_len=None,
    ):
        if self.vocoder_only:
            raise RuntimeError(
                "Text inference is disabled in vocoder_only mode. Use vocoder_infer with mel/spec input."
            )
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            g = None
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths, g=g)
        if self.use_sdp:
            logw = self.dp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)
        else:
            logw = self.dp(x, x_mask, g=g)
        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(
            x_mask.dtype
        )
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        o, _, _ = self.dec((z * y_mask)[:, :, :max_len], g=g)
        return o, attn, y_mask, (z, z_p, m_p, logs_p)

    def vocoder_infer(self, y, y_lengths, sid=None, max_len=None):
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)
        else:
            g = None
        
        if self.use_vae2_bottleneck:
            # VAE2 mode: mel -> latent -> upsample -> vocoder
            z_latent, mean, log_var, z_mask, z_lengths = self.vae2_encoder(y, y_lengths, g=g)
            # Use mean for inference (no sampling)
            z_upsampled, y_mask = self.vae2_upsampler(mean, z_mask, target_length=y.size(2), g=g)
            decoded = (z_upsampled * y_mask)[:, :, :max_len] if max_len is not None else (z_upsampled * y_mask)
            o, _, _ = self.dec(decoded, g=g)
            return o, y_mask, (z_latent, mean, log_var)
        else:
            z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
            decoded = (z * y_mask)[:, :, :max_len] if max_len is not None else (z * y_mask)
            o, _, _ = self.dec(decoded, g=g)
            return o, y_mask, (z, m_q, logs_q)
    
    def encode_to_latent(self, y, y_lengths, sid=None):
        """
        Encode mel spectrogram to latent space for flow matching training.
        
        Args:
            y: [B, mel_channels, T] mel spectrogram at 100Hz
            y_lengths: [B] lengths
            sid: speaker id (optional)
            
        Returns:
            latent: [B, latent_dim, T//vae2_downsample] latent (mean, deterministic)
            z_lengths: [B] latent lengths
        """
        if not self.use_vae2_bottleneck:
            raise RuntimeError("encode_to_latent requires use_vae2_bottleneck=True")
        
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)
        else:
            g = None
        
        z_latent, mean, log_var, z_mask, z_lengths = self.vae2_encoder(y, y_lengths, g=g)
        # Return mean (deterministic) for flow matching
        return mean * z_mask, z_lengths, z_mask
    
    def decode_from_latent(self, latent, latent_lengths, target_length=None, sid=None):
        """
        Decode from latent to audio.
        
        Args:
            latent: [B, latent_dim, T_z] latent features
            latent_lengths: [B] latent lengths
            target_length: target mel length at 100Hz (optional)
            sid: speaker id (optional)
            
        Returns:
            audio: [B, 1, T_audio] generated audio
        """
        if not self.use_vae2_bottleneck:
            raise RuntimeError("decode_from_latent requires use_vae2_bottleneck=True")
        
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)
        else:
            g = None
        
        # Create latent mask
        z_mask = torch.unsqueeze(commons.sequence_mask(latent_lengths, latent.size(2)), 1).to(latent.dtype)
        
        # Compute target length if not provided
        if target_length is None:
            target_length = latent.size(2) * self.vae2_downsample
        
        # Upsample
        z_upsampled, y_mask = self.vae2_upsampler(latent, z_mask, target_length=target_length, g=g)
        
        # Decode to audio
        o, _, _ = self.dec(z_upsampled * y_mask, g=g)
        return o, y_mask

    # currently vits-2 is not capable of voice conversion
    ## comment - choihkk
    ## Assuming the use of the ResidualCouplingTransformersLayer2 module, it seems that voice conversion is possible 
    def voice_conversion(self, y, y_lengths, sid_src, sid_tgt):
        assert self.n_speakers > 0, "n_speakers have to be larger than 0."
        g_src = self.emb_g(sid_src).unsqueeze(-1)
        g_tgt = self.emb_g(sid_tgt).unsqueeze(-1)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g_src)
        z_p = self.flow(z, y_mask, g=g_src)
        z_hat = self.flow(z_p, y_mask, g=g_tgt, reverse=True)
        o_hat = self.dec(z_hat * y_mask, g=g_tgt)
        return o_hat, y_mask, (z, z_p, z_hat)

class DiscriminatorCQT(nn.Module):
    def __init__(self, cfg, hop_length, n_octaves, bins_per_octave):
        super(DiscriminatorCQT, self).__init__()
        self.cfg = cfg

        self.filters = cfg.model.mssbcqtd.filters
        self.max_filters = cfg.model.mssbcqtd.max_filters
        self.filters_scale = cfg.model.mssbcqtd.filters_scale
        self.kernel_size = (3, 9)
        self.dilations = cfg.model.mssbcqtd.dilations
        self.stride = (1, 2)

        self.in_channels = cfg.model.mssbcqtd.in_channels
        self.out_channels = cfg.model.mssbcqtd.out_channels
        self.fs = cfg.model.mssbcqtd.sampling_rate
        self.hop_length = hop_length
        self.n_octaves = n_octaves
        self.bins_per_octave = bins_per_octave

        self.cqt_transform = features.cqt.CQT2010v2(
            sr=self.fs * 2,
            hop_length=self.hop_length,
            n_bins=self.bins_per_octave * self.n_octaves,
            bins_per_octave=self.bins_per_octave,
            output_format="Complex",
            pad_mode="constant",
        )

        self.conv_pres = nn.ModuleList()
        for i in range(self.n_octaves):
            self.conv_pres.append(
                NormConv2d(
                    self.in_channels * 2,
                    self.in_channels * 2,
                    kernel_size=self.kernel_size,
                    padding=get_2d_padding(self.kernel_size),
                )
            )

        self.convs = nn.ModuleList()

        self.convs.append(
            NormConv2d(
                self.in_channels * 2,
                self.filters,
                kernel_size=self.kernel_size,
                padding=get_2d_padding(self.kernel_size),
            )
        )

        in_chs = min(self.filters_scale * self.filters, self.max_filters)
        for i, dilation in enumerate(self.dilations):
            out_chs = min(
                (self.filters_scale ** (i + 1)) * self.filters, self.max_filters
            )
            self.convs.append(
                NormConv2d(
                    in_chs,
                    out_chs,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    dilation=(dilation, 1),
                    padding=get_2d_padding(self.kernel_size, (dilation, 1)),
                    norm="weight_norm",
                )
            )
            in_chs = out_chs
        out_chs = min(
            (self.filters_scale ** (len(self.dilations) + 1)) * self.filters,
            self.max_filters,
        )
        self.convs.append(
            NormConv2d(
                in_chs,
                out_chs,
                kernel_size=(self.kernel_size[0], self.kernel_size[0]),
                padding=get_2d_padding((self.kernel_size[0], self.kernel_size[0])),
                norm="weight_norm",
            )
        )

        self.conv_post = NormConv2d(
            out_chs,
            self.out_channels,
            kernel_size=(self.kernel_size[0], self.kernel_size[0]),
            padding=get_2d_padding((self.kernel_size[0], self.kernel_size[0])),
            norm="weight_norm",
        )

        self.activation = torch.nn.LeakyReLU(negative_slope=LRELU_SLOPE)
        self.resample = T.Resample(orig_freq=self.fs, new_freq=self.fs * 2)

    def forward(self, x):
        fmap = []

        x = self.resample(x)

        z = self.cqt_transform(x)

        z_amplitude = z[:, :, :, 0].unsqueeze(1)
        z_phase = z[:, :, :, 1].unsqueeze(1)

        z = torch.cat([z_amplitude, z_phase], dim=1)
        z = rearrange(z, "b c w t -> b c t w")

        latent_z = []
        for i in range(self.n_octaves):
            latent_z.append(
                self.conv_pres[i](
                    z[
                        :,
                        :,
                        :,
                        i * self.bins_per_octave : (i + 1) * self.bins_per_octave,
                    ]
                )
            )
        latent_z = torch.cat(latent_z, dim=-1)

        for i, l in enumerate(self.convs):
            latent_z = l(latent_z)

            latent_z = self.activation(latent_z)
            fmap.append(latent_z)

        latent_z = self.conv_post(latent_z)

        return latent_z, fmap


class MultiScaleSubbandCQTDiscriminator(nn.Module):
    def __init__(self, cfg):
        super(MultiScaleSubbandCQTDiscriminator, self).__init__()
        self.cfg = cfg
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorCQT(
                    cfg,
                    hop_length=cfg.model.mssbcqtd.hop_lengths[i],
                    n_octaves=cfg.model.mssbcqtd.n_octaves[i],
                    bins_per_octave=cfg.model.mssbcqtd.bins_per_octaves[i],
                )
                for i in range(len(cfg.model.mssbcqtd.hop_lengths))
            ]
        )

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for disc in self.discriminators:
            y_d_r, fmap_r = disc(y)
            y_d_g, fmap_g = disc(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
    
class DiscriminatorR(nn.Module):
    def __init__(self, cfg: AttrDict, resolution: List[List[int]]):
        super().__init__()

        self.resolution = resolution
        assert (
            len(self.resolution) == 3
        ), f"MRD layer requires list with len=3, got {self.resolution}"
        self.lrelu_slope = 0.1

        norm_f = weight_norm 
        self.d_mult = 1
        self.convs = nn.ModuleList(
            [
                norm_f(nn.Conv2d(1, int(32 * self.d_mult), (3, 9), padding=(1, 4))),
                norm_f(
                    nn.Conv2d(
                        int(32 * self.d_mult),
                        int(32 * self.d_mult),
                        (3, 9),
                        stride=(1, 2),
                        padding=(1, 4),
                    )
                ),
                norm_f(
                    nn.Conv2d(
                        int(32 * self.d_mult),
                        int(32 * self.d_mult),
                        (3, 9),
                        stride=(1, 2),
                        padding=(1, 4),
                    )
                ),
                norm_f(
                    nn.Conv2d(
                        int(32 * self.d_mult),
                        int(32 * self.d_mult),
                        (3, 9),
                        stride=(1, 2),
                        padding=(1, 4),
                    )
                ),
                norm_f(
                    nn.Conv2d(
                        int(32 * self.d_mult),
                        int(32 * self.d_mult),
                        (3, 3),
                        padding=(1, 1),
                    )
                ),
            ]
        )
        self.conv_post = norm_f(
            nn.Conv2d(int(32 * self.d_mult), 1, (3, 3), padding=(1, 1))
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []

        x = self.spectrogram(x)
        x = x.unsqueeze(1)
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, self.lrelu_slope)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

    def spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        n_fft, hop_length, win_length = self.resolution
        x = F.pad(
            x,
            (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)),
            mode="reflect",
        )
        x = x.squeeze(1)
        x = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            center=False,
            return_complex=True,
        )
        x = torch.view_as_real(x)  # [B, F, TT, 2]
        mag = torch.norm(x, p=2, dim=-1)  # [B, F, TT]

        return mag


class MultiResolutionDiscriminator(nn.Module):
    def __init__(self, cfg, debug=False):
        super().__init__()
        self.resolutions = cfg.resolutions
        assert (
            len(self.resolutions) == 3
        ), f"MRD requires list of list with len=3, each element having a list with len=3. Got {self.resolutions}"
        self.discriminators = nn.ModuleList(
            [DiscriminatorR(cfg, resolution) for resolution in self.resolutions]
        )

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[List[torch.Tensor]],
        List[List[torch.Tensor]],
    ]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(x=y)
            y_d_g, fmap_g = d(x=y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
    
class AMPBlock1(torch.nn.Module):
    """
    AMPBlock applies Snake / SnakeBeta activation functions with trainable parameters that control periodicity, defined for each layer.
    AMPBlock1 has additional self.convs2 that contains additional Conv1d layers with a fixed dilation=1 followed by each layer in self.convs1

    Args:
        h (AttrDict): Hyperparameters.
        channels (int): Number of convolution channels.
        kernel_size (int): Size of the convolution kernel. Default is 3.
        dilation (tuple): Dilation rates for the convolutions. Each dilation layer has two convolutions. Default is (1, 3, 5).
        activation (str): Activation function type. Should be either 'snake' or 'snakebeta'. Default is None.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple = (1, 3, 5),
        activation: str = None,
    ):
        super().__init__()

        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=d,
                        padding=get_padding(kernel_size, d),
                    )
                )
                for d in dilation
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                )
                for _ in range(len(dilation))
            ]
        )
        self.convs2.apply(init_weights)

        self.num_layers = len(self.convs1) + len(
            self.convs2
        )  # Total number of conv layers

        # Select which Activation1d, lazy-load cuda version to ensure backward compatibility
        Activation1d = TorchActivation1d

        # Activation functions
        if activation == "snakebeta":
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=SnakeBeta(
                            channels, alpha_logscale=True
                        )
                    )
                    for _ in range(self.num_layers)
                ]
            )
        else:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

    def forward(self, x):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x

        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)