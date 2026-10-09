from pathlib import Path
import subprocess
import json
import sys


# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"D:\OratorIQ")

METADATA = ROOT / "data" / "contrastive" / "metadata.jsonl"

INFERENCE_SCRIPT = (
    ROOT / "scripts" / "infer_oratoriq.py"
)

RESULT_FILE = (
    ROOT
    / "artifacts"
    / "results"
    / "inference_result.json"
)

REPORT_FILE = (
    ROOT
    / "artifacts"
    / "results"
    / "six_flaw_type_inference_report.json"
)


# ============================================================
# FLAW TYPES
# ============================================================

FLAW_TYPES = [
    "fast_pacing",
    "slow_pacing",
    "high_volume",
    "low_volume",
    "pitch_deviation",
    "long_pause",
]


# ============================================================
# LOAD METADATA
# ============================================================

def load_metadata():

    print()
    print("=" * 75)
    print("LOADING DATASET METADATA")
    print("=" * 75)

    if not METADATA.exists():

        raise FileNotFoundError(
            f"Metadata not found:\n{METADATA}"
        )

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

    print(
        f"Loaded metadata records: "
        f"{len(records):,}"
    )

    return records


# ============================================================
# FIND A VALID PAIR
# ============================================================

def find_pair(records, flaw_type):

    # Find flawed recordings for this flaw type
    flawed_candidates = [
        r for r in records
        if r.get("label") != "good"
        and r.get("flaw_type") == flaw_type
    ]

    if not flawed_candidates:
        return None, None

    # Try candidates until we find a matching good recording
    for flawed in flawed_candidates:

        pair_id = flawed.get("pair_id")

        good_candidates = [
            r for r in records
            if r.get("pair_id") == pair_id
            and r.get("label") == "good"
        ]

        if not good_candidates:
            continue

        good = good_candidates[0]

        good_path = ROOT / Path(
            good["audio"]
        )

        flawed_path = ROOT / Path(
            flawed["audio"]
        )

        if (
            good_path.exists()
            and flawed_path.exists()
        ):
            return good_path, flawed_path

    return None, None


# ============================================================
# RUN ONE INFERENCE
# ============================================================

def run_test(
    flaw_type,
    good_path,
    flawed_path
):

    print()
    print("=" * 75)
    print(
        f"TESTING: {flaw_type.upper()}"
    )
    print("=" * 75)

    print(
        f"Good:   {good_path}"
    )

    print(
        f"Flawed: {flawed_path}"
    )

    command = [
        sys.executable,
        "-X",
        "utf8",
        str(INFERENCE_SCRIPT),
        "--good",
        str(good_path),
        "--flawed",
        str(flawed_path),
    ]

    try:

        process = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace"
        )

    except Exception as e:

        return {
            "expected_flaw_type": flaw_type,
            "status": "ERROR",
            "error": str(e)
        }

    print(process.stdout)

    if process.returncode != 0:

        print(
            "Inference failed:"
        )

        print(
            process.stderr
        )

        return {
            "expected_flaw_type": flaw_type,
            "status": "ERROR",
            "error": process.stderr[-3000:]
        }

    if not RESULT_FILE.exists():

        return {
            "expected_flaw_type": flaw_type,
            "status": "ERROR",
            "error": (
                "inference_result.json "
                "was not created"
            )
        }

    try:

        with open(
            RESULT_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            result = json.load(f)

    except Exception as e:

        return {
            "expected_flaw_type": flaw_type,
            "status": "ERROR",
            "error": (
                "Could not read inference result: "
                + str(e)
            )
        }

    regions = result.get(
        "regions",
        []
    )

    detected_types = [
        region.get("flaw_type")
        for region in regions
    ]

    detected = (
        len(regions) > 0
    )

    correct_type = (
        flaw_type in detected_types
    )

    return {

        "expected_flaw_type":
            flaw_type,

        "status":
            "OK",

        "detected":
            detected,

        "correct_type":
            correct_type,

        "detected_regions":
            len(regions),

        "detected_types":
            detected_types,

        "regions":
            regions,

        "audio":
            str(flawed_path)
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print(
        "ORATORIQ — SIX FLAW TYPE "
        "INFERENCE TEST"
    )
    print("=" * 75)

    records = load_metadata()

    results = []

    for flaw_type in FLAW_TYPES:

        good_path, flawed_path = (
            find_pair(
                records,
                flaw_type
            )
        )

        if (
            good_path is None
            or flawed_path is None
        ):

            print()
            print(
                f"Could not find valid pair "
                f"for {flaw_type}"
            )

            results.append({

                "expected_flaw_type":
                    flaw_type,

                "status":
                    "NO_PAIR"
            })

            continue

        result = run_test(
            flaw_type,
            good_path,
            flawed_path
        )

        results.append(
            result
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 75)
    print("FINAL SIX-TYPE SUMMARY")
    print("=" * 75)

    detected_count = 0
    correct_count = 0

    for result in results:

        flaw_type = result[
            "expected_flaw_type"
        ]

        status = result.get(
            "status"
        )

        if status != "OK":

            print(
                f"{flaw_type:20s} "
                f"[ERROR] {status}"
            )

            continue

        detected = result.get(
            "detected",
            False
        )

        correct = result.get(
            "correct_type",
            False
        )

        if detected:
            detected_count += 1

        if correct:
            correct_count += 1

        if detected and correct:

            print(
                f"{flaw_type:20s} "
                f"[PASS] Detected + correct type"
            )

        elif detected:

            print(
                f"{flaw_type:20s} "
                f"[WARN] Detected but wrong type"
            )

            print(
                f"{'':20s}"
                f"Predicted: "
                f"{result.get('detected_types')}"
            )

        else:

            print(
                f"{flaw_type:20s} "
                f"[FAIL] No region detected"
            )

    # ========================================================
    # OVERALL
    # ========================================================

    print()
    print("=" * 75)
    print("OVERALL")
    print("=" * 75)

    print(
        f"Detected: "
        f"{detected_count}/6"
    )

    print(
        f"Correct flaw type: "
        f"{correct_count}/6"
    )

    # ========================================================
    # SAVE REPORT
    # ========================================================

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )

    print()
    print(
        f"Report saved to:\n"
        f"{REPORT_FILE}"
    )


if __name__ == "__main__":
    main()