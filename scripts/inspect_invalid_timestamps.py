import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
METADATA = PROJECT / "data" / "contrastive" / "metadata.jsonl"

print("=" * 80)
print("ORATORIQ — INVALID TIMESTAMP INSPECTOR")
print("=" * 80)

rows = []

with METADATA.open("r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))

invalid = []

for r in rows:
    if r.get("flaw_type") is None:
        continue

    start = r.get("flaw_start")
    end = r.get("flaw_end")
    duration = r.get("original_duration")

    reason = None

    if start is None:
        reason = "MISSING_START"

    elif end is None:
        reason = "MISSING_END"

    elif start < 0:
        reason = "NEGATIVE_START"

    elif end <= start:
        reason = "END_BEFORE_OR_EQUAL_START"

    elif duration is not None and end > duration:
        reason = "END_AFTER_AUDIO_DURATION"

    if reason:
        invalid.append((r, reason))

print()
print(f"Invalid records found: {len(invalid)}")
print()

for i, (r, reason) in enumerate(invalid, 1):

    print(f"[{i:02d}] {r.get('id')}")
    print(f"     pair_id:       {r.get('pair_id')}")
    print(f"     label:         {r.get('label')}")
    print(f"     flaw_type:     {r.get('flaw_type')}")
    print(f"     flaw_start:    {r.get('flaw_start')}")
    print(f"     flaw_end:      {r.get('flaw_end')}")
    print(f"     duration:      {r.get('original_duration')}")
    print(f"     severity:      {r.get('severity')}")
    print(f"     REASON:        {reason}")
    print()

print("=" * 80)

# Summary
from collections import Counter

reasons = Counter(reason for _, reason in invalid)

print("INVALID TIMESTAMP SUMMARY")
print("-" * 80)

for reason, count in reasons.items():
    print(f"{reason:<35} {count}")

print("=" * 80)