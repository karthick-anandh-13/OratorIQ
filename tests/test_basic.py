import os
import json
from pathlib import Path

import pytest

import speechlab.dataset as dataset_module
from speechlab.dataset import build_dataset
from speechlab.features import extract_features


def test_build_dataset(tmp_path: Path, monkeypatch):
    # Use a temporary data directory to avoid polluting repo
    data_root = tmp_path / "data"
    monkeypatch.setattr(dataset_module, "DATA_ROOT", data_root)
    monkeypatch.setattr(dataset_module, "GOOD_DIR", data_root / "good")
    monkeypatch.setattr(dataset_module, "BAD_DIR", data_root / "bad")
    monkeypatch.setattr(dataset_module, "ALIGN_DIR", data_root / "alignments")
    monkeypatch.setattr(dataset_module, "FLAW_LABELS_DIR", data_root / "flaw_labels")
    monkeypatch.setattr(dataset_module, "MANIFEST_PATH", data_root / "manifest.jsonl")
    for directory in (
        dataset_module.GOOD_DIR,
        dataset_module.BAD_DIR,
        dataset_module.ALIGN_DIR,
        dataset_module.FLAW_LABELS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    # Force rebuild
    build_dataset(force=True)
    manifest_path = dataset_module.MANIFEST_PATH
    assert manifest_path.exists(), "Manifest should be created"
    # Check at least one entry
    with open(manifest_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) > 0

def test_extract_features(tmp_path: Path, monkeypatch):
    # Setup a minimal manifest with a tiny wav file
    data_root = tmp_path / "data"
    good_dir = data_root / "good"
    good_dir.mkdir(parents=True, exist_ok=True)
    # Create a 1-second silent wav
    import numpy as np, soundfile as sf
    wav_path = good_dir / "silent.wav"
    sr = 16000
    sf.write(wav_path, np.zeros(sr), sr)
    # Create manifest entry
    manifest_path = data_root / "manifest.jsonl"
    entry = {
        "pair_id": "test",
        "speaker": "test",
        "source_type": "human",
        "license": "Public Domain",
        "source_url": "",
        "audio_path": str(wav_path),
        "transcript": "",
        "audio_hash": "dummyhash",
        "alignment_path": "",
        "severity": 0.0,
        "flaws": []
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(entry, f)
        f.write("\n")
    # Patch paths in module
    monkeypatch.setattr("speechlab.features.MANIFEST_PATH", manifest_path)
    out_dir = tmp_path / "features"
    extract_features(manifest_path, out_dir, overwrite=True)
    # Expect a .npz file
    npz_files = list(out_dir.glob("*.npz"))
    assert len(npz_files) == 1
    # Load and verify keys
    data = np.load(npz_files[0])
    assert "rms" in data and "mfcc" in data
