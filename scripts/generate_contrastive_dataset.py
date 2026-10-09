"""
OratorIQ - Contrastive Speech Dataset Generator

Reads:
    D:/OratorIQ/LibriSpeech/dev-clean

Creates:
    D:/OratorIQ/data/contrastive

Dataset levels:
    good
    slight
    medium
    bad
    extreme

The script creates controlled, reproducible acoustic delivery flaws
and records exact temporal ground-truth information.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
import soundfile as sf


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LIBRISPEECH_DIR = PROJECT_ROOT / "LibriSpeech" / "dev-clean"
OUTPUT_DIR = PROJECT_ROOT / "data" / "contrastive"

SEED = 42

LEVELS = [
    "good",
    "slight",
    "medium",
    "bad",
    "extreme",
]


# ============================================================
# RANDOMNESS
# ============================================================

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# DIRECTORY SETUP
# ============================================================

def create_directories() -> None:
    directories = [
        OUTPUT_DIR / "audio" / "good",
        OUTPUT_DIR / "audio" / "slight",
        OUTPUT_DIR / "audio" / "medium",
        OUTPUT_DIR / "audio" / "bad",
        OUTPUT_DIR / "audio" / "extreme",
        OUTPUT_DIR / "transcripts",
        OUTPUT_DIR / "labels",
        OUTPUT_DIR / "alignments",
        OUTPUT_DIR / "features",
        OUTPUT_DIR / "train",
        OUTPUT_DIR / "validation",
        OUTPUT_DIR / "test",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# TRANSCRIPT HANDLING
# ============================================================

def find_transcript_file(audio_path: Path) -> Path | None:
    """
    LibriSpeech stores transcripts approximately like:

        251-136532-0003.flac
        251-136532.trans.txt

    Both are in the same speaker directory.
    """

    speaker_dir = audio_path.parent

    candidates = list(speaker_dir.glob("*.trans.txt"))

    if not candidates:
        return None

    return candidates[0]


def load_transcript(audio_path: Path) -> str | None:
    transcript_file = find_transcript_file(audio_path)

    if transcript_file is None:
        return None

    target_id = audio_path.stem

    try:
        with transcript_file.open(
            "r",
            encoding="utf-8",
            errors="ignore",
        ) as file:

            for line in file:
                line = line.strip()

                if not line:
                    continue

                parts = line.split(maxsplit=1)

                if len(parts) != 2:
                    continue

                utterance_id, transcript = parts

                if utterance_id == target_id:
                    return transcript.strip()

    except Exception as exc:
        print(f"[WARNING] Could not read transcript: {transcript_file}")
        print(exc)

    return None


# ============================================================
# AUDIO UTILITIES
# ============================================================

def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Prevent clipping while keeping the waveform usable.
    """

    audio = np.asarray(audio, dtype=np.float32)

    peak = np.max(np.abs(audio))

    if peak > 0.99:
        audio = audio / peak * 0.98

    return audio


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio)) + 1e-12))


def apply_gain(audio: np.ndarray, gain_db: float) -> np.ndarray:
    gain = 10 ** (gain_db / 20.0)
    return audio * gain


def choose_flaw_region(
    duration: float,
    rng: random.Random,
) -> Tuple[float, float]:

    if duration < 2.5:
        return 0.2, max(0.5, duration - 0.2)

    # Avoid the very beginning/end.
    start_min = duration * 0.20
    start_max = duration * 0.55

    start = rng.uniform(start_min, start_max)

    region_length = rng.uniform(
        max(0.7, duration * 0.10),
        max(1.0, duration * 0.25),
    )

    end = min(start + region_length, duration * 0.85)

    if end <= start:
        end = min(duration, start + 0.5)

    return start, end


def split_audio(
    audio: np.ndarray,
    sr: int,
    start: float,
    end: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    start_sample = int(start * sr)
    end_sample = int(end * sr)

    before = audio[:start_sample]
    middle = audio[start_sample:end_sample]
    after = audio[end_sample:]

    return before, middle, after


# ============================================================
# FLAW GENERATORS
# ============================================================

def fast_pacing(
    audio: np.ndarray,
    sr: int,
    severity: str,
) -> Tuple[np.ndarray, Dict]:

    duration = len(audio) / sr
    rng = random.Random(SEED + len(audio))

    start, end = choose_flaw_region(duration, rng)

    before, middle, after = split_audio(
        audio,
        sr,
        start,
        end,
    )

    rates = {
        "slight": 1.10,
        "medium": 1.20,
        "bad": 1.35,
        "extreme": 1.50,
    }

    rate = rates[severity]

    modified = librosa.effects.time_stretch(
        middle.astype(np.float32),
        rate=rate,
    )

    output = np.concatenate(
        [before, modified, after]
    )

    new_end = start + len(modified) / sr

    label = {
        "flaw_type": "fast_pacing",
        "start": round(start, 4),
        "end": round(new_end, 4),
        "severity": severity,
        "parameter": {
            "time_stretch_rate": rate
        },
    }

    return normalize_audio(output), label


def slow_pacing(
    audio: np.ndarray,
    sr: int,
    severity: str,
) -> Tuple[np.ndarray, Dict]:

    duration = len(audio) / sr
    rng = random.Random(SEED + len(audio) + 1)

    start, end = choose_flaw_region(duration, rng)

    before, middle, after = split_audio(
        audio,
        sr,
        start,
        end,
    )

    rates = {
        "slight": 0.93,
        "medium": 0.85,
        "bad": 0.72,
        "extreme": 0.60,
    }

    rate = rates[severity]

    modified = librosa.effects.time_stretch(
        middle.astype(np.float32),
        rate=rate,
    )

    output = np.concatenate(
        [before, modified, after]
    )

    new_end = start + len(modified) / sr

    label = {
        "flaw_type": "slow_pacing",
        "start": round(start, 4),
        "end": round(new_end, 4),
        "severity": severity,
        "parameter": {
            "time_stretch_rate": rate
        },
    }

    return normalize_audio(output), label


def long_pause(
    audio: np.ndarray,
    sr: int,
    severity: str,
) -> Tuple[np.ndarray, Dict]:

    duration = len(audio) / sr
    rng = random.Random(SEED + len(audio) + 2)

    start, end = choose_flaw_region(duration, rng)

    before, middle, after = split_audio(
        audio,
        sr,
        start,
        end,
    )

    pause_lengths = {
        "slight": 0.30,
        "medium": 0.60,
        "bad": 1.00,
        "extreme": 1.80,
    }

    pause_duration = pause_lengths[severity]

    # Insert silence before the selected region.
    silence = np.zeros(
        int(pause_duration * sr),
        dtype=np.float32,
    )

    output = np.concatenate(
        [before, silence, middle, after]
    )

    flaw_start = start
    flaw_end = start + pause_duration

    label = {
        "flaw_type": "long_pause",
        "start": round(flaw_start, 4),
        "end": round(flaw_end, 4),
        "severity": severity,
        "parameter": {
            "pause_seconds": pause_duration
        },
    }

    return normalize_audio(output), label


def low_volume(
    audio: np.ndarray,
    sr: int,
    severity: str,
) -> Tuple[np.ndarray, Dict]:

    duration = len(audio) / sr
    rng = random.Random(SEED + len(audio) + 3)

    start, end = choose_flaw_region(duration, rng)

    before, middle, after = split_audio(
        audio,
        sr,
        start,
        end,
    )

    reductions = {
        "slight": -3.0,
        "medium": -6.0,
        "bad": -10.0,
        "extreme": -15.0,
    }

    reduction = reductions[severity]

    modified = apply_gain(
        middle,
        reduction,
    )

    output = np.concatenate(
        [before, modified, after]
    )

    label = {
        "flaw_type": "low_volume",
        "start": round(start, 4),
        "end": round(end, 4),
        "severity": severity,
        "parameter": {
            "gain_db": reduction
        },
    }

    return normalize_audio(output), label


def high_volume(
    audio: np.ndarray,
    sr: int,
    severity: str,
) -> Tuple[np.ndarray, Dict]:

    duration = len(audio) / sr
    rng = random.Random(SEED + len(audio) + 4)

    start, end = choose_flaw_region(duration, rng)

    before, middle, after = split_audio(
        audio,
        sr,
        start,
        end,
    )

    gains = {
        "slight": 2.0,
        "medium": 4.0,
        "bad": 7.0,
        "extreme": 10.0,
    }

    gain = gains[severity]

    modified = apply_gain(
        middle,
        gain,
    )

    output = np.concatenate(
        [before, modified, after]
    )

    label = {
        "flaw_type": "high_volume",
        "start": round(start, 4),
        "end": round(end, 4),
        "severity": severity,
        "parameter": {
            "gain_db": gain
        },
    }

    return normalize_audio(output), label


def pitch_shift(
    audio: np.ndarray,
    sr: int,
    severity: str,
) -> Tuple[np.ndarray, Dict]:

    duration = len(audio) / sr
    rng = random.Random(SEED + len(audio) + 5)

    start, end = choose_flaw_region(duration, rng)

    before, middle, after = split_audio(
        audio,
        sr,
        start,
        end,
    )

    shifts = {
        "slight": 1.0,
        "medium": 2.0,
        "bad": 3.5,
        "extreme": 5.0,
    }

    semitones = rng.choice(
        [-shifts[severity], shifts[severity]]
    )

    modified = librosa.effects.pitch_shift(
        middle.astype(np.float32),
        sr=sr,
        n_steps=semitones,
    )

    output = np.concatenate(
        [before, modified, after]
    )

    label = {
        "flaw_type": "pitch_deviation",
        "start": round(start, 4),
        "end": round(end, 4),
        "severity": severity,
        "parameter": {
            "pitch_shift_semitones": semitones
        },
    }

    return normalize_audio(output), label


# ============================================================
# GENERATE ONE FLAW
# ============================================================

def generate_flaw(
    audio: np.ndarray,
    sr: int,
    severity: str,
    flaw_type: str,
) -> Tuple[np.ndarray, Dict]:

    generators = {
        "fast_pacing": fast_pacing,
        "slow_pacing": slow_pacing,
        "long_pause": long_pause,
        "low_volume": low_volume,
        "high_volume": high_volume,
        "pitch_deviation": pitch_shift,
    }

    generator = generators[flaw_type]

    return generator(
        audio,
        sr,
        severity,
    )


# ============================================================
# MAIN PROCESSING
# ============================================================

def process_file(
    audio_path: Path,
    metadata_file,
    rng: random.Random,
) -> bool:

    transcript = load_transcript(audio_path)

    if not transcript:
        print(
            f"[SKIP] Transcript not found: {audio_path.name}"
        )
        return False

    try:
        audio, sr = librosa.load(
            audio_path,
            sr=16000,
            mono=True,
        )

    except Exception as exc:
        print(
            f"[ERROR] Could not load {audio_path}"
        )
        print(exc)
        return False

    if len(audio) < sr:
        print(
            f"[SKIP] Audio too short: {audio_path.name}"
        )
        return False

    audio = normalize_audio(audio)

    sample_id = audio_path.stem

    original_duration = len(audio) / sr

    # --------------------------------------------------------
    # GOOD
    # --------------------------------------------------------

    good_path = (
        OUTPUT_DIR
        / "audio"
        / "good"
        / f"{sample_id}.wav"
    )

    sf.write(
        good_path,
        audio,
        sr,
    )

    transcript_path = (
        OUTPUT_DIR
        / "transcripts"
        / f"{sample_id}.txt"
    )

    transcript_path.write_text(
        transcript,
        encoding="utf-8",
    )

    good_record = {
        "id": f"{sample_id}_good",
        "pair_id": sample_id,
        "audio": str(
            good_path.relative_to(PROJECT_ROOT)
        ),
        "transcript": transcript,
        "label": "good",
        "severity": 0.0,
        "flaw_type": None,
        "flaw_start": None,
        "flaw_end": None,
        "original_duration": round(
            original_duration,
            4,
        ),
    }

    metadata_file.write(
        json.dumps(
            good_record,
            ensure_ascii=False,
        )
        + "\n"
    )

    # --------------------------------------------------------
    # FLAWED LEVELS
    # --------------------------------------------------------

    severity_values = {
        "slight": 0.25,
        "medium": 0.50,
        "bad": 0.75,
        "extreme": 1.00,
    }

    # Randomly choose flaw types, but reproducibly.
    flaw_types = [
        "fast_pacing",
        "slow_pacing",
        "long_pause",
        "low_volume",
        "high_volume",
        "pitch_deviation",
    ]

    for level in [
        "slight",
        "medium",
        "bad",
        "extreme",
    ]:

        flaw_type = rng.choice(flaw_types)

        try:
            modified_audio, label = generate_flaw(
                audio.copy(),
                sr,
                level,
                flaw_type,
            )

        except Exception as exc:
            print(
                f"[ERROR] Failed {level} generation "
                f"for {sample_id}: {exc}"
            )
            continue

        output_path = (
            OUTPUT_DIR
            / "audio"
            / level
            / f"{sample_id}_{level}.wav"
        )

        sf.write(
            output_path,
            modified_audio,
            sr,
        )

        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        record = {
            "id": f"{sample_id}_{level}",
            "pair_id": sample_id,
            "audio": str(
                output_path.relative_to(PROJECT_ROOT)
            ),
            "original_audio": str(
                good_path.relative_to(PROJECT_ROOT)
            ),
            "transcript": transcript,
            "label": level,
            "severity": severity_values[level],
            "flaw_type": label["flaw_type"],
            "flaw_start": label["start"],
            "flaw_end": label["end"],
            "parameters": label["parameter"],
            "sample_rate": sr,
            "original_duration": round(
                original_duration,
                4,
            ),
            "modified_duration": round(
                len(modified_audio) / sr,
                4,
            ),
        }

        metadata_file.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )

        # ----------------------------------------------------
        # Separate label file
        # ----------------------------------------------------

        label_path = (
            OUTPUT_DIR
            / "labels"
            / f"{sample_id}_{level}.json"
        )

        label_path.write_text(
            json.dumps(
                record,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        # ----------------------------------------------------
        # Alignment file
        # ----------------------------------------------------

        alignment_path = (
            OUTPUT_DIR
            / "alignments"
            / f"{sample_id}_{level}.json"
        )

        alignment_data = {
            "pair_id": sample_id,
            "audio": str(
                output_path.relative_to(PROJECT_ROOT)
            ),
            "transcript": transcript,
            "flaw": {
                "type": label["flaw_type"],
                "start": label["start"],
                "end": label["end"],
            },
        }

        alignment_path.write_text(
            json.dumps(
                alignment_data,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    return True


# ============================================================
# DATASET SPLIT
# ============================================================

def create_splits(records: List[Dict]) -> None:
    """
    Split by pair_id rather than individual audio file.

    This prevents the same original speech from appearing
    in both train and test.
    """

    pair_ids = sorted(
        {
            record["pair_id"]
            for record in records
        }
    )

    random.Random(SEED).shuffle(pair_ids)

    total = len(pair_ids)

    train_end = int(total * 0.70)
    validation_end = int(total * 0.85)

    train_ids = set(
        pair_ids[:train_end]
    )

    validation_ids = set(
        pair_ids[train_end:validation_end]
    )

    test_ids = set(
        pair_ids[validation_end:]
    )

    split_map = {}

    for pair_id in train_ids:
        split_map[pair_id] = "train"

    for pair_id in validation_ids:
        split_map[pair_id] = "validation"

    for pair_id in test_ids:
        split_map[pair_id] = "test"

    split_files = {
        "train": OUTPUT_DIR / "train" / "metadata.jsonl",
        "validation": OUTPUT_DIR / "validation" / "metadata.jsonl",
        "test": OUTPUT_DIR / "test" / "metadata.jsonl",
    }

    handles = {
        name: path.open(
            "w",
            encoding="utf-8",
        )
        for name, path in split_files.items()
    }

    try:
        for record in records:

            split = split_map[
                record["pair_id"]
            ]

            handles[split].write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    finally:

        for handle in handles.values():
            handle.close()

    split_summary = {
        "train_pairs": len(train_ids),
        "validation_pairs": len(validation_ids),
        "test_pairs": len(test_ids),
        "total_pairs": total,
    }

    summary_path = (
        OUTPUT_DIR
        / "split_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            split_summary,
            indent=2,
        ),
        encoding="utf-8",
    )


# ============================================================
# ENTRY POINT
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Generate the OratorIQ contrastive "
            "speech dataset."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help=(
            "Maximum number of original LibriSpeech "
            "recordings to process. Default: 50"
        ),
    )

    args = parser.parse_args()

    print("=" * 70)
    print("ORATORIQ CONTRASTIVE DATASET GENERATOR")
    print("=" * 70)

    print()
    print(f"Input : {LIBRISPEECH_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Limit : {args.limit}")
    print()

    if not LIBRISPEECH_DIR.exists():

        raise FileNotFoundError(
            f"LibriSpeech directory not found:\n"
            f"{LIBRISPEECH_DIR}"
        )

    create_directories()

    audio_files = sorted(
        LIBRISPEECH_DIR.rglob("*.flac")
    )

    print(
        f"Found {len(audio_files)} FLAC files."
    )

    if not audio_files:
        print("No FLAC files found.")
        return

    selected_files = audio_files[
        : args.limit
    ]

    print(
        f"Processing {len(selected_files)} files..."
    )
    print()

    metadata_path = (
        OUTPUT_DIR
        / "metadata.jsonl"
    )

    records = []

    rng = random.Random(SEED)

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as metadata_file:

        for index, audio_path in enumerate(
            selected_files,
            start=1,
        ):

            print(
                f"[{index}/{len(selected_files)}] "
                f"{audio_path.name}"
            )

            success = process_file(
                audio_path,
                metadata_file,
                rng,
            )

            if success:
                # Reconstruct metadata for split generation.
                # Read later from the JSONL file.
                pass

    # --------------------------------------------------------
    # Read generated metadata
    # --------------------------------------------------------

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if line:
                records.append(
                    json.loads(line)
                )

    # --------------------------------------------------------
    # Create train/validation/test
    # --------------------------------------------------------

    create_splits(records)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {
        "input_directory": str(
            LIBRISPEECH_DIR
        ),
        "output_directory": str(
            OUTPUT_DIR
        ),
        "original_files_processed": len(
            selected_files
        ),
        "metadata_records": len(records),
        "levels": LEVELS,
        "random_seed": SEED,
    }

    summary_path = (
        OUTPUT_DIR
        / "dataset_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print("DATASET GENERATION COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Original recordings : {len(selected_files)}"
    )

    print(
        f"Generated records    : {len(records)}"
    )

    print()
    print(
        f"Dataset location:"
    )

    print(
        OUTPUT_DIR
    )

    print()
    print(
        "Next step: inspect the generated audio "
        "before increasing the dataset size."
    )


if __name__ == "__main__":
    main()