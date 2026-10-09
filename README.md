OratorIQ

OratorIQ is a speech-analytics application combining audio-based fluency analysis, contrastive delivery analysis, temporal event localization, and retrieval-augmented coaching. It provides a FastAPI backend and a React frontend for uploading speech recordings and viewing analysis results.

Important: OratorIQ is an AI-assisted demonstration and coaching tool, not a medical or clinical diagnostic system. Model outputs are predictions and candidate regions; they may be incorrect and should not be treated as a diagnosis.

Highlights

Standalone audio analysis — analyze supported speech recordings without requiring a matching clean-reference recording.

Contrastive analysis — compare a recording with a corresponding clean reference when one is available.

Fluency analysis — use the existing V4 inference pipeline to estimate fluency/stuttering-related probabilities and possible event categories.

Delivery analysis — use the existing hybrid/contrastive pipeline to identify candidate delivery issues and relevant time regions when required inputs are available.

Temporal grounding — return candidate timestamps for detected regions where supported by the analysis pipeline.

Local RAG coaching — retrieve guidance from the Markdown coaching knowledge base and produce observations, recommendations, and exercises using local TF-IDF retrieval.

Dashboard and API — FastAPI endpoints expose analysis, health, and dashboard functionality.

React interface — browser-based audio upload and result display.

Analysis modes

Standalone mode

Standalone mode is intended for recordings that do not have a matching clean reference. It uses existing audio-only inference components where compatible. Metrics that fundamentally require a reference comparison should be marked unavailable rather than estimated or fabricated.

Contrastive mode

Contrastive mode uses a recording and its corresponding clean reference when available. The existing contrastive pipeline is retained. Reference matching in the current dataset uses the recording's pair identifier and clean audio files available under the contrastive data directory.

The precise endpoints and request schemas are defined by the running API. Start the backend and open http://127.0.0.1:8001/docs to inspect available routes and upload parameters.

Architecture

React frontend (web/)
        |
        | HTTP requests
        v
FastAPI backend (api/main.py)
        |
        +--> Standalone audio inference
        |       +--> Existing V4 fluency inference
        |
        +--> Contrastive delivery-analysis pipeline
        |       +--> Reference audio / dataset metadata
        |       +--> Temporal, flaw-type and severity inference
        |
        +--> Local RAG coaching
                +--> Markdown knowledge base (rag/knowledge/)
                +--> TF-IDF retrieval and coaching

The components used depend on the selected analysis mode and the availability of required inputs.

Repository structure

OratorIQ/
├── api/
│   └── main.py                 # FastAPI application and API routes
├── artifacts/                  # Runtime outputs, uploads, generated artifacts
├── data/                       # Local datasets and contrastive reference audio
├── LibriSpeech/                # Local speech dataset, if used
├── models/                     # Model files used by inference
├── rag/
│   ├── knowledge/              # Coaching knowledge in Markdown
│   ├── storage/                # Local retrieval storage/index artifacts
│   ├── coach.py                # Coaching response generation
│   ├── ingest.py                # Knowledge ingestion and index building
│   └── retrieve.py             # Retrieval utilities
├── results/                    # Generated results, if used by local scripts
├── scripts/                    # Training/inference and pipeline scripts
├── src/                        # Supporting project source
├── tests/                      # Automated tests
├── web/
│   ├── src/
│   │   ├── App.jsx              # React application
│   │   ├── main.jsx             # Frontend entry point
│   │   └── styles.css           # Application styles
│   ├── package.json
│   └── package-lock.json
├── .gitignore
├── Makefile
├── pyproject.toml
├── requirements.txt
└── README.md

Large datasets, generated results, local environments, frontend dependencies, and model artifacts may be excluded from Git. Check the actual repository configuration before assuming every local model or dataset is present in a fresh clone.

Requirements

Windows, macOS, or Linux (the commands below use Windows PowerShell)

Python version compatible with requirements.txt and pyproject.toml

Node.js and npm

Git

Audio-decoding dependencies required by the Python packages

Required trained model files and any dataset/reference files needed by the selected analysis mode

Some model and dataset files may be large and may not be included in Git. Obtain them from the project's authorized source before running inference if they are not present locally.

Setup on Windows

1. Clone the repository

git clone https://github.com/karthick-anandh-13/OratorIQ.git
cd OratorIQ

2. Create and activate a Python environment

Use a Python version supported by the project's dependencies.

py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

If PowerShell blocks environment activation, use an appropriate local execution-policy setting or activate the environment through your IDE. Avoid disabling system security settings unnecessarily.

3. Install backend dependencies

pip install -r requirements.txt

If the project declares additional dependencies in pyproject.toml, follow that file's installation instructions as well.

4. Start the backend

From the repository root:

python -m uvicorn api.main:app --host 127.0.0.1 --port 8001

The API should be available at:

API base: http://127.0.0.1:8001

Interactive API documentation: http://127.0.0.1:8001/docs

OpenAPI schema: http://127.0.0.1:8001/openapi.json

Health check: http://127.0.0.1:8001/health

Keep the backend terminal running while using the frontend.

5. Install frontend dependencies

Open a second terminal:

cd D:\OratorIQ\web
npm ci

If you cloned the repository to a different location, change to that clone's web directory instead.

6. Start the frontend

Inspect web/package.json for the available scripts. For a typical Vite-based setup, run:

npm run dev

Use the local URL printed by the command. Ensure the frontend API base URL points to http://127.0.0.1:8001 for local development.

Running tests

From the repository root, with the Python environment activated:

python -m pytest -q

The maintainer reported 9 tests passing in a test run on October 9, 2026, including standalone audio analysis, contrastive analysis with a matching reference, health, and dashboard tests. Test results can change as the code and environment change; rerun the suite to verify the current checkout.

Warnings may still appear even when tests pass. Review model feature-name and dependency deprecation warnings rather than assuming they are harmless in every environment.

RAG coaching knowledge

The local coaching knowledge is stored as Markdown files under rag/knowledge/, including topics such as:

Fluency

Pacing

Pauses

Pitch

Volume

Blocking

Prolongation

Repetitions

Presentation coaching

rag/ingest.py builds local retrieval data from the knowledge files. Rebuild the index only when necessary and follow the implementation's instructions. Do not commit generated indexes or large artifacts unless the project explicitly requires them for deployment.

API exploration

With the backend running, visit:

http://127.0.0.1:8001/docs

Use the interactive docu
