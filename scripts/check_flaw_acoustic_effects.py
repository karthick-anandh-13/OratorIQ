from pathlib import Path
import json

import librosa
import numpy as np


ROOT = Path(r"D:\OratorIQ")

METADATA = (
    ROOT
    / "data"
    / "contrastive"
    / "metadata.jsonl"
)


TARGET_TYPES = [
    "fast_pacing",
    "slow_pacing",
    "high_volume",
    "low_volume",
    "pitch_deviation",
    "long_pause",
]


SR = 16000


def load_metadata():

    records = []

    with open(
        METADATA,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if line:
                records.append(
                    json.loads(line)
                )

    return records


def load_audio(path):

    y, sr = librosa.load(
        path,
        sr=SR,
        mono=True
    )

    return y, sr


def rms_db(y):

    rms = librosa.feature.rms(
        y=y,
        frame_length=min(
            1024,
            len(y)
        ),
        hop_length=256
    )[0]

    value = float(
        np.mean(rms)
    )

    return 20 * np.log10(
        max(value, 1e-8)
    )


def silence_ratio(y):

    intervals = librosa.effects.split(
        y,
        top_db=30
    )

    voiced = sum(
        max(0, end - start)
        for start, end in intervals
    )

    return 1.0 - (
        voiced / max(len(y), 1)
    )


def f0_mean(y):

    try:

        f0 = librosa.yin(
            y,
            fmin=70,
            fmax=400,
            sr=SR,
            frame_length=1024,
            hop_length=256
        )

        f0 = f0[
            np.isfinite(f0)
        ]

        if len(f0):
            return float(
                np.mean(f0)
            )

    except Exception:
        pass

    return 0.0


def analyze(path):

    y, sr = load_audio(path)

    duration = len(y) / sr

    rms = rms_db(y)

    silence = silence_ratio(y)

    f0 = f0_mean(y)

    return {
        "duration": duration,
        "rms_db": rms,
        "silence_ratio": silence,
        "f0": f0,
    }


def main():

    print("=" * 75)
    print("ORATORIQ — DATASET ACOUSTIC SANITY CHECK")
    print("=" * 75)

    records = load_metadata()

    print(
        f"Metadata records: {len(records)}"
    )

    for flaw_type in TARGET_TYPES:

        print()
        print("=" * 75)
        print(
            f"CHECKING: {flaw_type.upper()}"
        )
        print("=" * 75)

        flawed = next(
            r
            for r in records
            if r["label"] != "good"
            and r["flaw_type"] == flaw_type
        )

        pair_id = flawed["pair_id"]

        good = next(
            r
            for r in records
            if r["label"] == "good"
            and r["pair_id"] == pair_id
        )

        good_path = (
            ROOT / good["audio"]
        )

        flawed_path = (
            ROOT / flawed["audio"]
        )

        print()
        print(
            f"Good:   {good_path}"
        )

        print(
            f"Flawed: {flawed_path}"
        )

        print()

        print(
            f"Labeled flaw start: "
            f"{flawed['flaw_start']}"
        )

        print(
            f"Labeled flaw end:   "
            f"{flawed['flaw_end']}"
        )

        print(
            f"Severity: "
            f"{flawed['severity']}"
        )

        good_stats = analyze(
            good_path
        )

        flawed_stats = analyze(
            flawed_path
        )

        print()
        print("GOOD AUDIO")
        print(
            f"Duration:       "
            f"{good_stats['duration']:.3f}s"
        )

        print(
            f"RMS:            "
            f"{good_stats['rms_db']:.2f} dB"
        )

        print(
            f"Silence ratio:  "
            f"{good_stats['silence_ratio']:.3f}"
        )

        print(
            f"Mean F0:        "
            f"{good_stats['f0']:.2f} Hz"
        )

        print()
        print("FLAWED AUDIO")
        print(
            f"Duration:       "
            f"{flawed_stats['duration']:.3f}s"
        )

        print(
            f"RMS:            "
            f"{flawed_stats['rms_db']:.2f} dB"
        )

        print(
            f"Silence ratio:  "
            f"{flawed_stats['silence_ratio']:.3f}"
        )

        print(
            f"Mean F0:        "
            f"{flawed_stats['f0']:.2f} Hz"
        )

        print()
        print("CHANGE")

        print(
            f"Duration:       "
            f"{flawed_stats['duration'] - good_stats['duration']:+.3f}s"
        )

        print(
            f"RMS:            "
            f"{flawed_stats['rms_db'] - good_stats['rms_db']:+.2f} dB"
        )

        print(
            f"Silence ratio:  "
            f"{flawed_stats['silence_ratio'] - good_stats['silence_ratio']:+.3f}"
        )

        print(
            f"Mean F0:        "
            f"{flawed_stats['f0'] - good_stats['f0']:+.2f} Hz"
        )

    print()
    print("=" * 75)
    print("SANITY CHECK COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()