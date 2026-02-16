import os
import multiprocessing as mp

# --- 1. System Guardrails ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import os.path as osp
import numpy as np
import torch
import soundfile as sf
import librosa
import gc
import math
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from tqdm import tqdm

# --- Configuration ---
PAD_SAMPLES = 0  # Must match training - was 5000, now 0
SAMPLE_RATE = 44100
HOP_LENGTH = 441
WIN_LENGTH = 1764
N_FFT = 2048
N_MELS = 100
FMIN = 0
FMAX = None # None defaults to sr / 2
CENTER = False  # CRITICAL: Must match training code (center=False)

NUM_WORKERS = max(1, int(os.cpu_count()-1))

ROOT_DIR = "./"
TRAIN_LIST = "./train_list.txt"
VAL_LIST = "./val_list.txt"

TRAIN_OUTPUT_DIR = "./dump_100/train"
VAL_OUTPUT_DIR = "./dump_100/val"

# --- Normalization Functions ---

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

# --- GPU Functions (Run on Main Process) ---

def get_mel_gpu(wave_tensor, mel_basis, hann_window):
    """
    Runs on GPU using manual STFT to match training code EXACTLY.
    Uses center=False with manual reflect padding (same as mel_spectrogram_torch).
    """
    with torch.no_grad():
        # Manual reflect padding to match mel_spectrogram_torch with center=False
        # Padding amount: (n_fft - hop_length) / 2 on each side
        pad_amount = int((N_FFT - HOP_LENGTH) / 2)
        # wave_tensor shape: [B, T] -> [B, 1, T] for padding -> [B, T]
        wave_tensor = torch.nn.functional.pad(
            wave_tensor.unsqueeze(1),
            (pad_amount, pad_amount),
            mode='reflect'
        ).squeeze(1)
        
        # STFT with center=False (padding already applied manually)
        spec = torch.stft(
            wave_tensor,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window=hann_window,
            center=False,  # CRITICAL: must match training (was True)
            pad_mode='reflect',
            normalized=False,
            onesided=True,
            return_complex=False 
        )

        # Calculate Magnitude (Norm)
        spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)

        # Apply Mel Basis
        spec = torch.matmul(mel_basis, spec)

        # Spectral Normalization (Log)
        spec = spectral_normalize_torch(spec)

    return spec

# --- CPU Functions (Run on Workers) ---

def load_audio_cpu(path, target_sr):
    try:
        # sf.read automatically handles .flac, .wav, etc.
        wave, sr = sf.read(path)
        if wave.ndim > 1: wave = wave[:, 0]
        
        if not np.isfinite(wave).all():
            wave = np.nan_to_num(wave)

        if sr != target_sr:
            wave = librosa.resample(y=wave, orig_sr=sr, target_sr=target_sr)
            
        if PAD_SAMPLES > 0:
            wave = np.concatenate([np.zeros(PAD_SAMPLES), wave, np.zeros(PAD_SAMPLES)])
            
        return wave.astype(np.float32)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def cpu_worker_task(line, root_path):
    try:
        parts = line.strip().split('|')
        rel_path = parts[0]
        # Ignore speaker_id (parts[3]) as requested
        
        full_path = osp.join(root_path, rel_path)
        
        if not osp.exists(full_path):
            return None

        wave_np = load_audio_cpu(full_path, SAMPLE_RATE)
        if wave_np is None: return None

        # Return only path and wave
        return {
            "path": rel_path,
            "wave": wave_np
        }
    except Exception:
        return None

def run_pipeline(list_path, output_dir, desc):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{desc}] Pipeline starting. Processing on: {device}")

    mel_dir = osp.join(output_dir, "mels")
    os.makedirs(mel_dir, exist_ok=True)
    
    if not osp.exists(list_path):
        print(f"List file not found: {list_path}")
        return

    with open(list_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # --- Initialize Custom Mel Basis & Window on GPU ---
    # 1. Generate Mel Basis using Librosa
    mel_basis_np = librosa.filters.mel(
        sr=SAMPLE_RATE, 
        n_fft=N_FFT, 
        n_mels=N_MELS, 
        fmin=FMIN, 
        fmax=FMAX
    )
    # Convert to Torch and move to GPU
    mel_basis = torch.from_numpy(mel_basis_np).float().to(device)
    
    # 2. Create Hann Window for STFT
    hann_window = torch.hann_window(WIN_LENGTH).to(device)

    print(f"[{desc}] Processing files...")

    # --- Start CPU Workers ---
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        line_iterator = iter(lines)
        futures = set()
        
        full_queue_size = NUM_WORKERS * 4
        
        # Initial population of the queue
        for _ in range(full_queue_size):
            try:
                line = next(line_iterator)
                futures.add(executor.submit(cpu_worker_task, line, ROOT_DIR))
            except StopIteration:
                break
        
        with tqdm(total=len(lines), desc=desc) as pbar:
            processed_count = 0
            while futures:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                
                for future in done:
                    pbar.update(1)
                    processed_count += 1
                    
                    try:
                        result = future.result()
                        if result is not None:
                            # --- GPU Processing ---
                            wave_np = result['wave']
                            wave_tensor = torch.from_numpy(wave_np).unsqueeze(0).to(device)
                            
                            # Calculate Mel Spectrogram
                            mel = get_mel_gpu(wave_tensor, mel_basis, hann_window)
                            
                            # Ensure clean naming
                            safe_name = osp.splitext(result['path'])[0].replace("/", "_").replace("\\", "_") + ".pt"
                            mel_save_path = osp.join(mel_dir, safe_name)
                            
                            # Save ONLY Mel
                            data_to_save = {
                                "mel": mel.half().cpu()
                            }
                            
                            torch.save(data_to_save, mel_save_path)
                            del wave_tensor, mel, data_to_save
                            
                    except Exception as e:
                        print(f"\nError processing file: {e}")

                    # Add next task
                    try:
                        line = next(line_iterator)
                        futures.add(executor.submit(cpu_worker_task, line, ROOT_DIR))
                    except StopIteration:
                        pass
                
                if processed_count % 1000 == 0:
                    gc.collect()

def main():
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    #if os.path.exists(TRAIN_LIST):
    #    run_pipeline(TRAIN_LIST, TRAIN_OUTPUT_DIR, "Train")
    
    if os.path.exists(VAL_LIST):
        run_pipeline(VAL_LIST, VAL_OUTPUT_DIR, "Val")
        
    print("Feature extraction complete.")

if __name__ == "__main__":
    main()