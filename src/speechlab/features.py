import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import soundfile as sf
import pandas as pd

logger = logging.getLogger(__name__)

# Global deterministic seed is set in __init__.py
MANIFEST_PATH = Path(__file__).resolve().parents[2] / "data" / "manifest.jsonl"


def extract_features_from_wav(wav_path: Path) -> Dict[str, Any]:
    """Extract a minimal set of placeholder DSP features from a WAV file.

    This function is deterministic and does not depend on heavy external libraries.
    It returns a dictionary with feature arrays that can be saved to Parquet.
    """
    y, sr = sf.read(wav_path)
    # Simple frame-level RMS energy
    frame_len = int(0.025 * sr)  # 25 ms
    hop_len = int(0.010 * sr)   # 10 ms
    rms = []
    for start in range(0, len(y) - frame_len + 1, hop_len):
        frame = y[start:start + frame_len]
        rms.append(np.sqrt(np.mean(frame ** 2)))
    rms = np.array(rms)
    # Placeholder MFCC using numpy FFT magnitude (not real MFCC)
    fft_mag = np.abs(np.fft.rfft(y, n=512))
    mfcc = np.log1p(fft_mag)[:13]  # first 13 coefficients
    return {
        "rms": rms.tolist(),
        "mfcc": mfcc.tolist(),
        "sample_rate": sr,
    }

def extract_features(manifest_path: Path, output_dir: Path, overwrite: bool = False) -> None:
    """Iterate over the dataset manifest and extract features for each entry.

    Features are cached as ``.npz`` files named by the audio hash.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        entry: Dict[str, Any] = json.loads(line)
        audio_path = Path(entry["audio_path"])
        audio_hash = entry.get("audio_hash") or "unknown"
        out_file = output_dir / f"{audio_hash}.npz"
        if out_file.exists() and not overwrite:
            logger.info(f"Features for {audio_path} already exist – skipping.")
            continue
        feats = extract_features_from_wav(audio_path)
        np.savez_compressed(out_file, **feats)
        logger.info(f"Extracted features to {out_file}")
