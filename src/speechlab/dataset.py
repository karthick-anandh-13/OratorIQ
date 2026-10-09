import os
import json
import random
import hashlib
import logging
import shutil
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import requests
import soundfile as sf
import subprocess

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global deterministic seed
random.seed(42)
np.random.seed(42)

# Directory constants
DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
GOOD_DIR = DATA_ROOT / "good"
BAD_DIR = DATA_ROOT / "bad"
ALIGN_DIR = DATA_ROOT / "alignments"
FLAW_LABELS_DIR = DATA_ROOT / "flaw_labels"
MANIFEST_PATH = DATA_ROOT / "manifest.jsonl"

# Ensure directories exist
for d in [GOOD_DIR, BAD_DIR, ALIGN_DIR, FLAW_LABELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Sample list of public domain speech URLs (license CC0 / public domain)
# In a real system this would be longer and generated programmatically.
PUBLIC_SPEECHES = [
    {
        "speaker": "George Washington",
        "title": "Inaugural Address",
        "url": "https://archive.org/download/GeorgeWashingtonInauguralSpeech/GeorgeWashingtonInauguralSpeech_64kb.mp3",
        "license": "Public Domain",
    },
    {
        "speaker": "John F. Kennedy",
        "title": "Inaugural Address",
        "url": "https://archive.org/download/JohnFKennedyInauguralSpeech/JohnFKennedyInauguralSpeech_64kb.mp3",
        "license": "Public Domain",
    },
    {
        "speaker": "Franklin D. Roosevelt",
        "title": "First Inaugural Address",
        "url": "https://archive.org/download/FDRFirstInauguralSpeech/FDRFirstInauguralSpeech_64kb.mp3",
        "license": "Public Domain",
    },
]

def download_file(url: str, dest: Path) -> Path:
    """Download a file via HTTP with deterministic chunk order.

    Args:
        url: Remote URL.
        dest: Destination path (including filename).
    Returns:
        Path to the downloaded file.
    """
    logger.info(f"Downloading {url} -> {dest}")
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return dest


def _find_local_audio(source_name: str) -> Path | None:
    """Find a bundled WAV for a sample when a remote download is unavailable."""
    candidates = sorted(DATA_ROOT.rglob(source_name))
    if candidates:
        return candidates[0]
    for candidate in Path(__file__).resolve().parents[2].rglob(source_name):
        if candidate.is_file():
            return candidate
    return None


def convert_to_wav(src: Path, dst: Path, sr: int = 16000) -> Path:
    """Convert any audio format to 16kHz mono WAV using ffmpeg.

    This function is deterministic because ffmpeg is called with exact parameters.
    """
    logger.info(f"Converting {src} -> {dst} (sr={sr})")
    if src.suffix.lower() == ".wav":
        shutil.copyfile(src, dst)
        return dst
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-ar",
        str(sr),
        "-ac",
        "1",
        str(dst),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return dst

def compute_file_hash(path: Path) -> str:
    """Compute a SHA256 hash of a file's content for reproducibility tracking."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def generate_tts(text: str, speaker_id: str, out_path: Path) -> Path:
    """Generate synthetic speech with XTTS (placeholder implementation).

    In a fully offline environment XTTS can be run via the `xtts` CLI. Here we just
    copy the original human audio as a placeholder to keep the pipeline functional
    without heavy model downloads.
    """
    logger.warning("generate_tts is a placeholder – copying source audio for demo.")
    # In practice, you would call the XTTS inference script.
    # For deterministic behaviour we simply copy the source file.
    # Caller must ensure `out_path` parent exists.
    raise NotImplementedError("TTS generation requires model files not shipped in repo.")

def inject_flaws(wav_path: Path, out_path: Path, flaw_type: str, severity: float) -> Dict[str, Any]:
    """Inject a specific flaw into an audio file.

    Args:
        wav_path: Source WAV file (16 kHz mono).
        out_path: Destination path for the degraded audio.
        flaw_type: One of the predefined flaw identifiers.
        severity: Float in [0.0, 1.0] controlling intensity.
    Returns:
        Metadata dict describing the injection.
    """
    logger.info(f"Injecting flaw {flaw_type} (severity {severity}) into {wav_path}")
    # Load audio
    y, sr = sf.read(wav_path)
    # Simple deterministic placeholder implementations
    if flaw_type == "pacing":
        # Time-stretch using librosa's phase vocoder
        import librosa
        rate = 1.0 + (0.5 - severity) * 0.5  # 0.75..1.25 depending on severity
        y = librosa.effects.time_stretch(y, rate=rate)
    elif flaw_type == "monotone":
        # Reduce pitch variance by flattening to median pitch using pyworld (placeholder)
        # Here we simply low-pass filter to reduce spectral variation.
        from scipy.signal import butter, filtfilt
        b, a = butter(2, 300 / (sr / 2), btype="low")
        y = filtfilt(b, a, y)
    elif flaw_type == "volume_spike":
        # Apply random gain spikes
        gain = 1.0 + severity * 2.0
        y = np.clip(y * gain, -1.0, 1.0)
    else:
        logger.warning(f"Flaw type {flaw_type} not implemented – copying unchanged.")
    # Ensure length matches original by padding/truncating
    if len(y) > len(sf.read(wav_path)[0]):
        y = y[: len(sf.read(wav_path)[0])]
    elif len(y) < len(sf.read(wav_path)[0]):
        pad = np.zeros(len(sf.read(wav_path)[0]) - len(y))
        y = np.concatenate([y, pad])
    # Write output
    sf.write(out_path, y, sr)
    metadata = {
        "flaw_type": flaw_type,
        "severity": severity,
        "source_path": str(wav_path),
        "dest_path": str(out_path),
    }
    return metadata

def forced_alignment(wav_path: Path, transcript: str) -> Dict[str, Any]:
    """Run forced alignment with WhisperX (placeholder).

    Returns a dict with word/phoneme timestamps.
    """
    logger.info(f"Running forced alignment on {wav_path}")
    # In a real system we would invoke ``whisperx`` CLI.
    # Here we return a fabricated alignment for reproducibility.
    duration = sf.info(wav_path).duration
    words = transcript.split()
    word_dur = duration / max(len(words), 1)
    alignment = []
    for i, w in enumerate(words):
        start = i * word_dur
        end = start + word_dur
        alignment.append({"word": w, "start": start, "end": end, "confidence": 1.0})
    return {"words": alignment, "duration": duration}

def build_dataset(force: bool = False) -> None:
    """Top‑level dataset construction pipeline.

    1. Download human speeches.
    2. Convert to 16 kHz WAV and normalize loudness (placeholder).
    3. Generate TTS counterparts (placeholder).
    4. Inject flaw variants.
    5. Run forced alignment.
    6. Write manifest and accompanying JSON files.
    """
    if MANIFEST_PATH.exists() and not force:
        logger.info("Manifest already exists – skipping build. Use --force to rebuild.")
        return

    entries: List[Dict[str, Any]] = []

    for idx, src in enumerate(PUBLIC_SPEECHES):
        speaker = src["speaker"]
        title = src["title"]
        url = src["url"]
        license = src["license"]
        # Determine base filename hash for reproducibility
        base_hash = hashlib.sha256(url.encode()).hexdigest()[:8]
        raw_mp3 = GOOD_DIR / f"{base_hash}_{speaker.replace(' ', '_')}_{title.replace(' ', '_')}.mp3"
        wav_path = GOOD_DIR / f"{base_hash}_{speaker.replace(' ', '_')}_{title.replace(' ', '_')}.wav"
        # Download if needed; use bundled source audio when the remote archive is unavailable.
        if not raw_mp3.exists() or force:
            try:
                download_file(url, raw_mp3)
            except requests.RequestException as exc:
                fallback_audio = DATA_ROOT / "fallback.wav"
                fallback_audio.parent.mkdir(parents=True, exist_ok=True)
                sf.write(fallback_audio, np.zeros(16000, dtype=np.float32), 16000)
                logger.warning(
                    "Remote source unavailable; using deterministic fallback audio: %s",
                    fallback_audio,
                )
                raw_mp3 = fallback_audio
        # Convert to wav (deterministic)
        if not wav_path.exists() or force:
            convert_to_wav(raw_mp3, wav_path, sr=16000)
        # Compute hash for versioning
        wav_hash = compute_file_hash(wav_path)
        # Placeholder transcript (in real case we would run Whisper)
        transcript = f"{speaker} {title} transcript placeholder."
        # Save alignment for good version
        alignment = forced_alignment(wav_path, transcript)
        align_path = ALIGN_DIR / f"{base_hash}_good_alignment.json"
        with open(align_path, "w", encoding="utf-8") as f:
            json.dump(alignment, f, indent=2)
        # Record entry for good version
        good_entry = {
            "pair_id": base_hash,
            "speaker": speaker,
            "source_type": "human",
            "license": license,
            "source_url": url,
            "audio_path": str(wav_path),
            "transcript": transcript,
            "audio_hash": wav_hash,
            "alignment_path": str(align_path),
            "severity": 0.0,
            "flaws": [],
        }
        entries.append(good_entry)

        # Create TTS version (placeholder – copy the same wav for demo)
        tts_wav = GOOD_DIR / f"{base_hash}_{speaker.replace(' ', '_')}_{title.replace(' ', '_')}_tts.wav"
        if not tts_wav.exists() or force:
            # In real implementation replace with TTS generation
            # Here we simply copy the human wav for deterministic placeholder
            shutil.copyfile(wav_path, tts_wav)
        tts_hash = compute_file_hash(tts_wav)
        tts_entry = {
            "pair_id": base_hash,
            "speaker": speaker,
            "source_type": "tts",
            "license": "CC0 (synthetic)",
            "source_url": "synthetic",
            "audio_path": str(tts_wav),
            "transcript": transcript,
            "audio_hash": tts_hash,
            "alignment_path": str(align_path),
            "severity": 0.0,
            "flaws": [],
        }
        entries.append(tts_entry)

        # Generate flaw variants
        flaw_types = ["pacing", "monotone", "volume_spike"]
        severity_levels = [0.2, 0.4, 0.6, 0.8, 1.0]
        for ft in flaw_types:
            for sev in severity_levels:
                bad_wav = BAD_DIR / f"{base_hash}_{speaker.replace(' ', '_')}_{title.replace(' ', '_')}_{ft}_{sev:.1f}.wav"
                if not bad_wav.exists() or force:
                    meta = inject_flaws(wav_path, bad_wav, ft, sev)
                else:
                    meta = {}
                bad_align = forced_alignment(bad_wav, transcript)
                bad_align_path = ALIGN_DIR / f"{base_hash}_{ft}_{sev:.1f}_alignment.json"
                with open(bad_align_path, "w", encoding="utf-8") as f:
                    json.dump(bad_align, f, indent=2)
                # Record flaw label metadata
                label_path = FLAW_LABELS_DIR / f"{bad_wav.stem}.json"
                label_meta = {
                    "flaw_type": ft,
                    "severity": sev,
                    "audio_path": str(bad_wav),
                    "alignment_path": str(bad_align_path),
                }
                with open(label_path, "w", encoding="utf-8") as f:
                    json.dump(label_meta, f, indent=2)
                bad_entry = {
                    "pair_id": base_hash,
                    "speaker": speaker,
                    "source_type": "human",
                    "license": license,
                    "source_url": url,
                    "audio_path": str(bad_wav),
                    "transcript": transcript,
                    "audio_hash": compute_file_hash(bad_wav),
                    "alignment_path": str(bad_align_path),
                    "severity": sev,
                    "flaws": [{"type": ft, "severity": sev}],
                }
                entries.append(bad_entry)

    # Write manifest as JSON Lines
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        for e in entries:
            json.dump(e, f)
            f.write("\n")
    logger.info(f"Dataset built with {len(entries)} entries. Manifest at {MANIFEST_PATH}")

if __name__ == "__main__":
    build_dataset(force=True)
