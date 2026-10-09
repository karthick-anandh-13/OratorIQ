import json
import shutil
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]

METADATA = (
    PROJECT
    / "data"
    / "contrastive"
    / "metadata.jsonl"
)

BACKUP = (
    PROJECT
    / "data"
    / "contrastive"
    / "metadata_before_timestamp_repair.jsonl"
)

TEMP = (
    PROJECT
    / "data"
    / "contrastive"
    / "metadata_repaired.jsonl"
)


print("=" * 70)
print("ORATORIQ — TIMESTAMP REPAIR")
print("=" * 70)

print()
print(f"Input : {METADATA}")
print(f"Backup: {BACKUP}")
print()

# ------------------------------------------------------------
# Safety backup
# ------------------------------------------------------------

if not BACKUP.exists():
    shutil.copy2(
        METADATA,
        BACKUP
    )

    print("✓ Original metadata backed up.")

else:
    print("✓ Backup already exists.")

# ------------------------------------------------------------
# Read + repair
# ------------------------------------------------------------

total = 0
changed = 0

with METADATA.open(
    "r",
    encoding="utf-8"
) as source, TEMP.open(
    "w",
    encoding="utf-8"
) as output:

    for line in source:

        if not line.strip():
            continue

        record = json.loads(line)

        total += 1

        flaw_start = record.get(
            "flaw_start"
        )

        flaw_end = record.get(
            "flaw_end"
        )

        duration = record.get(
            "original_duration"
        )

        # Only repair flawed records.
        if (
            flaw_start is not None
            and flaw_end is not None
            and duration is not None
            and flaw_end > duration
        ):

            old_end = flaw_end

            # Clamp the timestamp to the actual audio.
            record["flaw_end"] = round(
                duration,
                4
            )

            # Keep an audit trail.
            record[
                "timestamp_repair"
            ] = {
                "original_flaw_end": old_end,
                "repaired_flaw_end": round(
                    duration,
                    4
                ),
                "reason": (
                    "flaw_end exceeded "
                    "original_audio_duration"
                ),
            }

            changed += 1

            print(
                f"[REPAIR {changed:02d}] "
                f"{record.get('id')}"
            )

            print(
                f"    {old_end:.4f}s "
                f"-> "
                f"{duration:.4f}s"
            )

        output.write(
            json.dumps(
                record,
                ensure_ascii=False
            )
            + "\n"
        )

# ------------------------------------------------------------
# Replace metadata
# ------------------------------------------------------------

shutil.move(
    str(TEMP),
    str(METADATA)
)

print()
print("=" * 70)
print("REPAIR COMPLETE")
print("=" * 70)

print(
    f"Records checked : {total:,}"
)

print(
    f"Records repaired: {changed:,}"
)

print()
print(
    f"Backup preserved at:"
)

print(
    BACKUP
)

print()
print(
    "The original audio files were NOT changed."
)

print("=" * 70)