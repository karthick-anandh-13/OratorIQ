# SpeechLab

SpeechLab is a local speech-analysis project for building contrastive datasets, extracting deterministic DSP features, running temporal-grounding analysis, and evaluating degraded speech variants.

## Features

- Python 3.11+ command-line workflow for dataset construction, feature extraction, analysis, and evaluation.
- FastAPI service for dashboard statistics and uploaded-audio analysis.
- Deterministic feature extraction using NumPy and SoundFile.
- React-compatible API endpoints for a future dashboard.

## Quick start

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m uvicorn api.main:app --host 127.0.0.1 --port 8001
```

Open http://127.0.0.1:8001/health to verify the service.

## Uploaded-audio analysis API

All analysis endpoints accept a multipart form field named `audio`.

- `POST /api/analyze` preserves the existing contrastive analysis behavior and requires a matching clean reference.
- `POST /api/analyze/standalone` analyzes WAV, FLAC, MP3, or OGG audio without a reference. Uploads are limited to 25 MiB and five minutes; silent, too-short, unsupported, and undecodable files are rejected.
- `POST /api/analyze/auto` uses contrastive analysis when a matching clean reference exists and otherwise selects standalone analysis. The React application uses this endpoint.

Standalone responses contain V4 audio-only fluency predictions and local coaching when findings are available. Reference-dependent delivery comparisons and event timestamps are explicitly unavailable in this mode. Speech presence is not independently verified, and model predictions are not a diagnosis.

## Commands

```powershell
speechlab --version
speechlab build_dataset --force
speechlab extract_features --overwrite
speechlab analyze path\to\audio.wav
speechlab evaluate --output results
```

The dataset builder downloads public-domain sample material and may require network access. Feature and evaluation commands require an existing manifest or generated dataset.

## Tests

```powershell
python -m pytest -q
```
