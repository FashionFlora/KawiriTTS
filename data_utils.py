import os
import random
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torch.utils.data

import commons
from preprocess.mel_processing import (mel_spectrogram_torch, spec_to_mel_torch,
                            spectrogram_torch)
from text import cleaned_text_to_sequence, text_to_sequence
from utils import load_filepaths_and_text, load_wav_to_torch


class TextAudioLoader(torch.utils.data.Dataset):
    """
    1) loads audio, text pairs
    2) normalizes text and converts them to sequences of integers
    3) computes spectrograms from audio files.
    """

    def __init__(self, audiopaths_and_text, hparams):
        self.hparams = hparams
        self.filelist_path = audiopaths_and_text
        self.filelist_dir = os.path.dirname(os.path.abspath(self.filelist_path))
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.sampling_rate = hparams.sampling_rate
        self.pad_samples = getattr(hparams, "pad_samples", 0)

        self.use_mel_spec_posterior = getattr(
            hparams, "use_mel_posterior_encoder", False
        )
        if self.use_mel_spec_posterior:
            self.n_mel_channels = getattr(hparams, "n_mel_channels", 80)
        self.cleaned_text = getattr(hparams, "cleaned_text", False)

        self.add_blank = hparams.add_blank
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 190)
        self.precomputed_mel_root = getattr(hparams, "precomputed_mel_root", "")
        self.precomputed_mel_subdir = getattr(hparams, "precomputed_mel_subdir", "mels")

        filelist_basename = os.path.basename(self.filelist_path).lower()
        if "train" in filelist_basename:
            self.filelist_split = "train"
        elif "val" in filelist_basename or "valid" in filelist_basename:
            self.filelist_split = "val"
        else:
            self.filelist_split = ""

        random.seed(1234)
        random.shuffle(self.audiopaths_and_text)
        self._filter()

    def _extract_audiopath_text(self, row):
        audiopath = row[0]
        text = row[-1] if len(row) > 1 else ""
        return audiopath, text

    def _resolve_audio_path(self, audiopath):
        if os.path.isabs(audiopath):
            return audiopath
        if os.path.isfile(audiopath):
            return audiopath
        
        # Try joining with filelist_dir
        path_from_dir = os.path.join(self.filelist_dir, audiopath)
        if os.path.isfile(path_from_dir):
            return path_from_dir
            
        # Try stripping ../ prefix if it exists and checking relative to filelist_dir
        if audiopath.startswith("../"):
            stripped_path = audiopath
            while stripped_path.startswith("../"):
                stripped_path = stripped_path[3:]
            
            path_stripped = os.path.join(self.filelist_dir, stripped_path)
            if os.path.isfile(path_stripped):
                return path_stripped
        
        return path_from_dir

    def _estimate_spec_length(self, audiopath):
        try:
            return max(1, sf.info(audiopath).frames // self.hop_length)
        except Exception:
            return max(1, os.path.getsize(audiopath) // (2 * self.hop_length))

    def _safe_mel_name(self, filename):
        normalized_filename = filename
        if os.path.isabs(filename):
            try:
                normalized_filename = os.path.relpath(filename, self.filelist_dir)
            except ValueError:
                normalized_filename = filename
        rel = os.path.splitext(normalized_filename)[0]
        return rel.replace("/", "_").replace("\\", "_") + ".pt"

    def _resolve_precomputed_mel_path(self, filename):
        if not self.precomputed_mel_root:
            return None

        candidate_splits = [self.filelist_split] if self.filelist_split else ["train", "val"]
        mel_name = self._safe_mel_name(filename)

        for split in candidate_splits:
            mel_path = os.path.join(
                self.precomputed_mel_root,
                split,
                self.precomputed_mel_subdir,
                mel_name,
            )
            if os.path.exists(mel_path):
                return mel_path
        return None

    def _load_saved_spec(self, spec_path):
        spec = torch.load(spec_path)
        if isinstance(spec, dict):
            if "mel" in spec:
                spec = spec["mel"]
            elif "spec" in spec:
                spec = spec["spec"]
            else:
                first_key = next(iter(spec.keys()))
                spec = spec[first_key]
        if spec.dim() == 3 and spec.size(0) == 1:
            spec = spec.squeeze(0)
        return spec.float()

    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length

        audiopaths_and_text_new = []
        lengths = []
        for row in self.audiopaths_and_text:
            audiopath, text = self._extract_audiopath_text(row)
            audiopath = self._resolve_audio_path(audiopath)
            if not os.path.isfile(audiopath):
                continue
            if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
                audiopaths_and_text_new.append([audiopath, text])
                lengths.append(self._estimate_spec_length(audiopath))
        self.audiopaths_and_text = audiopaths_and_text_new
        self.lengths = lengths

    def get_audio_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        text = self.get_text(text)
        spec, wav = self.get_audio(audiopath)
        return (text, spec, wav)

    def get_audio(self, filename):
        # TODO : if linear spec exists convert to mel from existing linear spec
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError(
                "{} {} SR doesn't match target {} SR".format(
                    sampling_rate, self.sampling_rate
                )
            )
        peak = torch.max(torch.abs(audio))
        if peak > 1.5:
            audio_norm = audio / self.max_wav_value
        else:
            audio_norm = audio

        if self.pad_samples > 0:
            audio_norm = F.pad(audio_norm, (self.pad_samples, self.pad_samples))

        audio_norm = audio_norm.unsqueeze(0)
        stem = os.path.splitext(filename)[0]
        spec_filename = stem + ".spec.pt"
        if self.use_mel_spec_posterior:
            spec_filename = spec_filename.replace(".spec.pt", ".mel.pt")

        precomputed_mel_path = self._resolve_precomputed_mel_path(filename)
        if self.use_mel_spec_posterior and precomputed_mel_path is not None:
            spec = self._load_saved_spec(precomputed_mel_path)
        elif os.path.exists(spec_filename):
            spec = self._load_saved_spec(spec_filename)
        else:
            if self.use_mel_spec_posterior:
                """TODO : (need verification)
                if linear spec exists convert to
                mel from existing linear spec (uncomment below lines)"""
                # if os.path.exists(filename.replace(".wav", ".spec.pt")):
                #     # spec, n_fft, num_mels, sampling_rate, fmin, fmax
                #     spec = spec_to_mel_torch(
                #         torch.load(filename.replace(".wav", ".spec.pt")),
                #         self.filter_length, self.n_mel_channels, self.sampling_rate,
                #         self.hparams.mel_fmin, self.hparams.mel_fmax)
                spec = mel_spectrogram_torch(
                    audio_norm,
                    self.filter_length,
                    self.n_mel_channels,
                    self.sampling_rate,
                    self.hop_length,
                    self.win_length,
                    self.hparams.mel_fmin,
                    self.hparams.mel_fmax,
                    center=False,
                )
            else:
                spec = spectrogram_torch(
                    audio_norm,
                    self.filter_length,
                    self.sampling_rate,
                    self.hop_length,
                    self.win_length,
                    center=False,
                )
            spec = torch.squeeze(spec, 0)
            try:
                torch.save(spec, spec_filename)
            except (OSError, IOError, RuntimeError) as e:
                # Don't crash worker if saving fails (disk full, permission issues, etc)
                pass
        return spec, audio_norm

    def get_text(self, text):
        text_norm = cleaned_text_to_sequence(text)
        if self.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def __getitem__(self, index):
        return self.get_audio_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextAudioCollate:
    """Zero-pads model inputs and targets"""

    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text and aduio
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]), dim=0, descending=True
        )

        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))

        text_padded = torch.LongTensor(len(batch), max_text_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        text_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row[0]
            text_padded[i, : text.size(0)] = text
            text_lengths[i] = text.size(0)

            spec = row[1]
            spec_padded[i, :, : spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, : wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

        if self.return_ids:
            return (
                text_padded,
                text_lengths,
                spec_padded,
                spec_lengths,
                wav_padded,
                wav_lengths,
                ids_sorted_decreasing,
            )
        return (
            text_padded,
            text_lengths,
            spec_padded,
            spec_lengths,
            wav_padded,
            wav_lengths,
        )


"""Multi speaker version"""


class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
    1) loads audio, speaker_id, text pairs
    2) normalizes text and converts them to sequences of integers
    3) computes spectrograms from audio files.
    """

    def __init__(self, audiopaths_sid_text, hparams):
        self.hparams = hparams
        self.filelist_path = audiopaths_sid_text
        self.filelist_dir = os.path.dirname(os.path.abspath(self.filelist_path))
        self.audiopaths_sid_text = load_filepaths_and_text(audiopaths_sid_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.sampling_rate = hparams.sampling_rate
        self.pad_samples = getattr(hparams, "pad_samples", 0)

        self.use_mel_spec_posterior = getattr(
            hparams, "use_mel_posterior_encoder", False
        )
        if self.use_mel_spec_posterior:
            self.n_mel_channels = getattr(hparams, "n_mel_channels", 80)
        self.cleaned_text = getattr(hparams, "cleaned_text", False)

        self.add_blank = hparams.add_blank
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 190)
        self.min_audio_len = getattr(hparams, "min_audio_len", 8192)
        self.precomputed_mel_root = getattr(hparams, "precomputed_mel_root", "")
        self.precomputed_mel_subdir = getattr(hparams, "precomputed_mel_subdir", "mels")

        filelist_basename = os.path.basename(self.filelist_path).lower()
        if "train" in filelist_basename:
            self.filelist_split = "train"
        elif "val" in filelist_basename or "valid" in filelist_basename:
            self.filelist_split = "val"
        else:
            self.filelist_split = ""

        random.seed(1234)
        random.shuffle(self.audiopaths_sid_text)
        self._filter()

    def _extract_audiopath_sid_text(self, row):
        audiopath = row[0]
        if len(row) >= 5:
            sid = row[3]
            text = row[4]
        elif len(row) >= 3:
            sid = row[1]
            text = row[2]
        else:
            raise ValueError(f"Invalid speaker filelist row: {row}")
        return audiopath, sid, text

    def _resolve_audio_path(self, audiopath):
        if os.path.isabs(audiopath):
            return audiopath
        if os.path.isfile(audiopath):
            return audiopath
        return os.path.join(self.filelist_dir, audiopath)

    def _estimate_spec_length(self, audiopath):
        try:
            return max(1, sf.info(audiopath).frames // self.hop_length)
        except Exception:
            return max(1, os.path.getsize(audiopath) // (2 * self.hop_length))

    def _safe_mel_name(self, filename):
        normalized_filename = filename
        if os.path.isabs(filename):
            try:
                normalized_filename = os.path.relpath(filename, self.filelist_dir)
            except ValueError:
                normalized_filename = filename
        rel = os.path.splitext(normalized_filename)[0]
        return rel.replace("/", "_").replace("\\", "_") + ".pt"

    def _resolve_precomputed_mel_path(self, filename):
        if not self.precomputed_mel_root:
            return None

        candidate_splits = [self.filelist_split] if self.filelist_split else ["train", "val"]
        mel_name = self._safe_mel_name(filename)

        for split in candidate_splits:
            mel_path = os.path.join(
                self.precomputed_mel_root,
                split,
                self.precomputed_mel_subdir,
                mel_name,
            )
            if os.path.exists(mel_path):
                return mel_path
        return None

    def _load_saved_spec(self, spec_path):
        spec = torch.load(spec_path)
        if isinstance(spec, dict):
            if "mel" in spec:
                spec = spec["mel"]
            elif "spec" in spec:
                spec = spec["spec"]
            else:
                first_key = next(iter(spec.keys()))
                spec = spec[first_key]
        if spec.dim() == 3 and spec.size(0) == 1:
            spec = spec.squeeze(0)
        return spec.float()

    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length

        audiopaths_sid_text_new = []
        lengths = []
        for row in self.audiopaths_sid_text:
            audiopath, sid, text = self._extract_audiopath_sid_text(row)
            audiopath = self._resolve_audio_path(audiopath)
            if not os.path.isfile(audiopath):
                continue
            if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
                length = self._estimate_spec_length(audiopath)
                if length < self.min_audio_len // self.hop_length:
                    continue
                audiopaths_sid_text_new.append([audiopath, sid, text])
                lengths.append(length)
        self.audiopaths_sid_text = audiopaths_sid_text_new
        self.lengths = lengths
        print(
            len(self.lengths)
        )  # if we use large corpus dataset, we can check how much time it takes.

    def get_audio_text_speaker_pair(self, audiopath_sid_text):
        # separate filename, speaker_id and text
        audiopath, sid, text = (
            audiopath_sid_text[0],
            audiopath_sid_text[1],
            audiopath_sid_text[2],
        )
        text = self.get_text(text)
        spec, wav = self.get_audio(audiopath)
        sid = self.get_sid(sid)
        return (text, spec, wav, sid)

    def get_audio(self, filename):
        # TODO : if linear spec exists convert to mel from existing linear spec
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError(
                "{} {} SR doesn't match target {} SR".format(
                    sampling_rate, self.sampling_rate
                )
            )
        peak = torch.max(torch.abs(audio))
        if peak > 1.5:
            audio_norm = audio / self.max_wav_value
        else:
            audio_norm = audio

        if self.pad_samples > 0:
            audio_norm = F.pad(audio_norm, (self.pad_samples, self.pad_samples))

        audio_norm = audio_norm.unsqueeze(0)
        stem = os.path.splitext(filename)[0]
        spec_filename = stem + ".spec.pt"
        if self.use_mel_spec_posterior:
            spec_filename = spec_filename.replace(".spec.pt", ".mel.pt")

        precomputed_mel_path = self._resolve_precomputed_mel_path(filename)
        if self.use_mel_spec_posterior and precomputed_mel_path is not None:
            spec = self._load_saved_spec(precomputed_mel_path)
        elif os.path.exists(spec_filename):
            spec = self._load_saved_spec(spec_filename)
        else:
            if self.use_mel_spec_posterior:
                """TODO : (need verification)
                if linear spec exists convert to
                mel from existing linear spec (uncomment below lines)"""
                # if os.path.exists(filename.replace(".wav", ".spec.pt")):
                #     # spec, n_fft, num_mels, sampling_rate, fmin, fmax
                #     spec = spec_to_mel_torch(
                #         torch.load(filename.replace(".wav", ".spec.pt")),
                #         self.filter_length, self.n_mel_channels, self.sampling_rate,
                #         self.hparams.mel_fmin, self.hparams.mel_fmax)
                spec = mel_spectrogram_torch(
                    audio_norm,
                    self.filter_length,
                    self.n_mel_channels,
                    self.sampling_rate,
                    self.hop_length,
                    self.win_length,
                    self.hparams.mel_fmin,
                    self.hparams.mel_fmax,
                    center=False,
                )
            else:
                spec = spectrogram_torch(
                    audio_norm,
                    self.filter_length,
                    self.sampling_rate,
                    self.hop_length,
                    self.win_length,
                    center=False,
                )
            spec = torch.squeeze(spec, 0)
            try:
                torch.save(spec, spec_filename)
            except (OSError, IOError, RuntimeError) as e:
                # Don't crash worker if saving fails (disk full, permission issues, etc)
                pass
        return spec, audio_norm

    def get_text(self, text):
        text_norm = cleaned_text_to_sequence(text)
        if self.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def get_sid(self, sid):
        sid = torch.LongTensor([int(sid)])
        return sid

    def __getitem__(self, index):
        return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index])

    def __len__(self):
        return len(self.audiopaths_sid_text)


class TextAudioSpeakerCollate:
    """Zero-pads model inputs and targets"""

    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text, audio and speaker identities
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized, sid]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]), dim=0, descending=True
        )

        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        sid = torch.LongTensor(len(batch))

        text_padded = torch.LongTensor(len(batch), max_text_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        text_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row[0]
            text_padded[i, : text.size(0)] = text
            text_lengths[i] = text.size(0)

            spec = row[1]
            spec_padded[i, :, : spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, : wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            sid[i] = row[3]

        if self.return_ids:
            return (
                text_padded,
                text_lengths,
                spec_padded,
                spec_lengths,
                wav_padded,
                wav_lengths,
                sid,
                ids_sorted_decreasing,
            )
        return (
            text_padded,
            text_lengths,
            spec_padded,
            spec_lengths,
            wav_padded,
            wav_lengths,
            sid,
        )


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.

    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        boundaries,
        num_replicas=None,
        rank=None,
        shuffle=True,
    ):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries

        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)

        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i + 1)

        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (
                total_batch_size - (len_bucket % total_batch_size)
            ) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]

            # add extra samples to make it evenly divisible
            rem = num_samples_bucket - len_bucket
            ids_bucket = (
                ids_bucket
                + ids_bucket * (rem // len_bucket)
                + ids_bucket[: (rem % len_bucket)]
            )

            # subsample
            ids_bucket = ids_bucket[self.rank :: self.num_replicas]

            # batching
            for j in range(len(ids_bucket) // self.batch_size):
                batch = [
                    bucket[idx]
                    for idx in ids_bucket[
                        j * self.batch_size : (j + 1) * self.batch_size
                    ]
                ]
                batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches

        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)

    def _bisect(self, x, lo=0, hi=None):
        if hi is None:
            hi = len(self.boundaries) - 1

        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self):
        return self.num_samples // self.batch_size
