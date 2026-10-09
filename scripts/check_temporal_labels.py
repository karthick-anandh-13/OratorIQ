import json
from pathlib import Path
from collections import Counter

PROJECT = Path(__file__).resolve().parents[1]
METADATA = PROJECT / "data" / "contrastive" / "metadata.jsonl"

print("=" * 70)
print("ORATORIQ — TEMPORAL LABEL AUDIT")
print("=" * 70)
print(f"Metadata: {METADATA}")
print()

rows = []

with METADATA.open("r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))

print(f"Total records: {len(rows):,}")
print()

# ------------------------------------------------------------
# Check flaw timestamps
# ------------------------------------------------------------

flawed = [r for r in rows if r.get("flaw_type") is not None]
good = [r for r in rows if r.get("flaw_type") is None]

missing_start = []
missing_end = []
invalid = []
valid = []

for r in flawed:
    start = r.get("flaw_start")
    end = r.get("flaw_end")
    duration = r.get("original_duration")

    if start is None:
        missing_start.append(r)
        continue

    if end is None:
        missing_end.append(r)
        continue

    if start < 0 or end <= start:
        invalid.append(r)
        continue

    if duration is not None and end > duration:
        invalid.append(r)
        continue

    valid.append(r)

print("TIMESTAMP VALIDATION")
print("-" * 70)
print(f"Good records:             {len(good):,}")
print(f"Flawed records:           {len(flawed):,}")
print(f"Valid flaw timestamps:    {len(valid):,}")
print(f"Missing flaw_start:       {len(missing_start):,}")
print(f"Missing flaw_end:         {len(missing_end):,}")
print(f"Invalid timestamps:       {len(invalid):,}")
print()

# ------------------------------------------------------------
# Flaw type distribution
# ------------------------------------------------------------

print("FLAW TYPE DISTRIBUTION")
print("-" * 70)

counts = Counter(r.get("flaw_type") for r in flawed)

for flaw_type, count in counts.most_common():
    print(f"{flaw_type:<20} {count:>6,}")

print()

# ------------------------------------------------------------
# Severity distribution
# ------------------------------------------------------------

print("SEVERITY DISTRIBUTION")
print("-" * 70)

severity_counts = Counter(r.get("label") for r in rows)

for label, count in severity_counts.items():
    print(f"{label:<12} {count:>6,}")

print()

# ------------------------------------------------------------
# Show examples
# ------------------------------------------------------------

print("EXAMPLE FLAW RECORDS")
print("-" * 70)

examples = valid[:10]

for r in examples:
    print(
        f"{r['id']:<35} "
        f"{r['label']:<8} "
        f"{r['flaw_type']:<18} "
        f"{r['flaw_start']:.3f}s -> "
        f"{r['flaw_end']:.3f}s"
    )

print()

# ------------------------------------------------------------
# Final verdict
# ------------------------------------------------------------

if len(valid) == len(flawed):
    print("✅ ALL FLAWED RECORDS HAVE VALID TEMPORAL LABELS")
    print()
    print("We can now build the temporal training dataset.")
else:
    print("⚠️ SOME RECORDS HAVE INVALID/MISSING TIMESTAMPS")
    print()
    print("Fix the timestamp generation before full training.")

print("=" * 70)