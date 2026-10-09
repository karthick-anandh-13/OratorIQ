import json
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA = PROJECT_ROOT / "data" / "contrastive" / "metadata.jsonl"
INFERENCE_SCRIPT = PROJECT_ROOT / "scripts" / "infer_oratoriq.py"


TARGET_TYPES = [
    "fast_pacing",
    "slow_pacing",
    "high_volume",
    "low_volume",
    "pitch_deviation",
    "long_pause",
]


def load_metadata():
    records = []

    with open(METADATA, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records


def find_pair(records, flaw_type):
    flawed = next(
        r for r in records
        if r["label"] != "good"
        and r["flaw_type"] == flaw_type
    )

    pair_id = flawed["pair_id"]

    good = next(
        r for r in records
        if r["label"] == "good"
        and r["pair_id"] == pair_id
    )

    return good, flawed


def run_inference(good_path, flawed_path):
    cmd = [
        sys.executable,
        "-X",
        "utf8",
        str(INFERENCE_SCRIPT),
        "--good",
        str(good_path),
        "--flawed",
        str(flawed_path),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    return result.stdout


def extract_result(output):
    lines = output.splitlines()

    actual_lines = []
    predicted_lines = []
    confidence_lines = []

    for line in lines:
        if "Flaw:" in line:
            predicted_lines.append(line.strip())

        if "Confidence:" in line:
            confidence_lines.append(line.strip())

    return predicted_lines, confidence_lines


def main():

    print("=" * 75)
    print("ORATORIQ — FLAW TYPE DIAGNOSTIC")
    print("=" * 75)

    records = load_metadata()

    print(f"Loaded metadata records: {len(records)}")

    for flaw_type in TARGET_TYPES:

        print("\n" + "=" * 75)
        print(f"TESTING: {flaw_type.upper()}")
        print("=" * 75)

        good, flawed = find_pair(records, flaw_type)

        good_path = PROJECT_ROOT / good["audio"]
        flawed_path = PROJECT_ROOT / flawed["audio"]

        print(f"Actual flaw: {flaw_type}")
        print(f"Good:       {good_path}")
        print(f"Flawed:     {flawed_path}")

        output = run_inference(good_path, flawed_path)

        predicted, confidence = extract_result(output)

        print("\nInference result:")

        if predicted:
            for line in predicted:
                print("  " + line)

        if confidence:
            for line in confidence:
                print("  " + line)

        if not predicted:
            print("  [WARNING] No detected flaw region.")

    print("\n" + "=" * 75)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()