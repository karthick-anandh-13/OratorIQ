from pathlib import Path
import pandas as pd
import re
import json


# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"D:\OratorIQ")

RAW_REPO = ROOT / "data" / "stutter" / "raw" / "ml-stuttering-events-dataset-main"
CLIPS_DIR = ROOT / "data" / "stutter" / "clips"

OUTPUT_DIR = ROOT / "data" / "stutter" / "metadata"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "stutter_metadata.csv"
OUTPUT_JSON = OUTPUT_DIR / "stutter_dataset_summary.json"


# ============================================================
# CONFIG
# ============================================================

LABEL_FILES = [
    RAW_REPO / "SEP-28k_labels.csv",
    RAW_REPO / "fluencybank_labels.csv",
]

STUTTER_LABELS = [
    "Prolongation",
    "Block",
    "SoundRep",
    "WordRep",
    "Interjection",
]

EXCLUDE_LABELS = [
    "Unsure",
    "PoorAudioQuality",
    "DifficultToUnderstand",
    "Music",
    "NoSpeech",
]


# ============================================================
# HELPERS
# ============================================================

def normalize_name(name):
    return str(name).strip().lower()


def parse_clip_filename(path):
    """
    Expected examples:

    HeStutters_0_0.wav
    FluencyBank_10_0.wav

    Returns:
        show, episode_id, clip_id
    """

    stem = path.stem

    match = re.match(r"^(.+)_([0-9]+)_([0-9]+)$", stem)

    if not match:
        return None

    show = match.group(1)
    episode_id = int(match.group(2))
    clip_id = int(match.group(3))

    return show, episode_id, clip_id


def get_primary_event(row):
    """
    Select the strongest stuttering event from the
    five stuttering-event annotations.
    """

    scores = {
        label: int(row.get(label, 0))
        for label in STUTTER_LABELS
    }

    max_score = max(scores.values())

    if max_score <= 0:
        if int(row.get("NoStutteredWords", 0)) > 0:
            return "normal"
        return "unknown"

    winners = [
        label
        for label, score in scores.items()
        if score == max_score
    ]

    # If multiple events have the same maximum score,
    # preserve them instead of arbitrarily selecting one.
    if len(winners) > 1:
        return "mixed"

    return winners[0]


def has_stutter(row):
    return any(
        int(row.get(label, 0)) > 0
        for label in STUTTER_LABELS
    )


# ============================================================
# LOAD LABEL DATA
# ============================================================

print("=" * 70)
print("ORATORIQ STUTTER DATASET PREPARATION")
print("=" * 70)

all_labels = []

for label_file in LABEL_FILES:

    if not label_file.exists():
        print(f"[WARNING] Missing label file: {label_file}")
        continue

    print(f"\n[LOAD] {label_file.name}")

    df = pd.read_csv(label_file)

    print(f"       Rows: {len(df):,}")

    dataset_name = label_file.stem.replace("_labels", "")

    df["dataset"] = dataset_name

    all_labels.append(df)


if not all_labels:
    raise RuntimeError("No label files found.")


labels = pd.concat(
    all_labels,
    ignore_index=True
)


print(f"\n[OK] Combined labels: {len(labels):,}")


# ============================================================
# CREATE LOOKUP TABLE
# ============================================================

labels["Show_norm"] = labels["Show"].map(normalize_name)

labels["key"] = (
    labels["Show_norm"]
    + "|"
    + labels["EpId"].astype(str)
    + "|"
    + labels["ClipId"].astype(str)
)


# Check duplicate label keys

duplicate_count = labels["key"].duplicated().sum()

print(f"[CHECK] Duplicate label keys: {duplicate_count:,}")

if duplicate_count > 0:
    print("[WARNING] Duplicate annotation keys detected.")


label_lookup = {}

for _, row in labels.iterrows():

    key = row["key"]

    # Keep first occurrence if duplicates exist.
    if key not in label_lookup:
        label_lookup[key] = row


# ============================================================
# FIND AUDIO CLIPS
# ============================================================

print("\n[SCAN] Searching extracted WAV files...")

audio_files = list(
    CLIPS_DIR.rglob("*.wav")
)

print(f"[OK] WAV clips found: {len(audio_files):,}")


# ============================================================
# MATCH AUDIO → LABEL
# ============================================================

records = []

matched = 0
unmatched = 0
invalid_filename = 0


for index, audio_path in enumerate(audio_files, start=1):

    parsed = parse_clip_filename(audio_path)

    if parsed is None:

        invalid_filename += 1
        continue

    show, episode_id, clip_id = parsed

    key = (
        normalize_name(show)
        + "|"
        + str(episode_id)
        + "|"
        + str(clip_id)
    )

    row = label_lookup.get(key)

    if row is None:

        unmatched += 1
        continue

    matched += 1

    record = {
        "audio_path": str(audio_path),
        "filename": audio_path.name,
        "show": show,
        "episode_id": episode_id,
        "clip_id": clip_id,
        "dataset": row["dataset"],
        "start": float(row["Start"]),
        "stop": float(row["Stop"]),
    }

    # Preserve all annotation values.
    for column in STUTTER_LABELS:
        record[column] = int(row[column])

    for column in [
        "Unsure",
        "PoorAudioQuality",
        "DifficultToUnderstand",
        "NoStutteredWords",
        "NaturalPause",
        "Music",
        "NoSpeech",
    ]:
        record[column] = int(row[column])

    record["has_stutter"] = int(has_stutter(row))

    record["primary_event"] = get_primary_event(row)

    records.append(record)

    if index % 1000 == 0:

        print(
            f"[PROGRESS] {index:,}/{len(audio_files):,} "
            f"| matched={matched:,} "
            f"| unmatched={unmatched:,}"
        )


# ============================================================
# CREATE DATAFRAME
# ============================================================

metadata = pd.DataFrame(records)


if metadata.empty:
    raise RuntimeError(
        "No audio clips could be matched with labels."
    )


# ============================================================
# SAVE METADATA
# ============================================================

metadata.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# DATASET STATISTICS
# ============================================================

event_distribution = (
    metadata["primary_event"]
    .value_counts()
    .to_dict()
)

stutter_distribution = (
    metadata["has_stutter"]
    .value_counts()
    .to_dict()
)

dataset_distribution = (
    metadata["dataset"]
    .value_counts()
    .to_dict()
)

summary = {
    "audio_files_found": len(audio_files),
    "matched": matched,
    "unmatched": unmatched,
    "invalid_filename": invalid_filename,
    "metadata_rows": len(metadata),
    "datasets": dataset_distribution,
    "stutter_distribution": {
        str(k): int(v)
        for k, v in stutter_distribution.items()
    },
    "primary_event_distribution": {
        str(k): int(v)
        for k, v in event_distribution.items()
    },
    "output_csv": str(OUTPUT_CSV),
}


with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("DATASET PREPARATION COMPLETE")
print("=" * 70)

print(f"Audio files found       : {len(audio_files):,}")
print(f"Matched with labels     : {matched:,}")
print(f"Unmatched               : {unmatched:,}")
print(f"Invalid filenames       : {invalid_filename:,}")
print(f"Final metadata rows     : {len(metadata):,}")

print("\nDataset distribution:")

for name, count in dataset_distribution.items():
    print(f"  {name:<20} {count:,}")

print("\nStutter distribution:")

for name, count in stutter_distribution.items():
    print(f"  {name:<20} {count:,}")

print("\nPrimary event distribution:")

for name, count in event_distribution.items():
    print(f"  {name:<25} {count:,}")

print("\nOutputs:")
print(f"  {OUTPUT_CSV}")
print(f"  {OUTPUT_JSON}")

print("=" * 70)