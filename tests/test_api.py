import json
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

import api.main as api
from api.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_dashboard_endpoint_returns_project_summary(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"pair_id": "sample", "speaker": "Test Speaker", "severity": 0.25}),
        encoding="utf-8",
    )
    monkeypatch.setattr("api.main.MANIFEST_PATH", manifest)

    response = client.get("/api/dashboard")

    assert response.status_code == 200
    payload = response.json()
    assert payload["entries"] == 1
    assert payload["degraded_entries"] == 1
    assert payload["average_severity"] == 0.25


def test_standalone_endpoint_processes_audio_without_pair_name(tmp_path, monkeypatch):
    audio_path = tmp_path / "sample.wav"
    sf.write(audio_path, np.linspace(-0.1, 0.1, 16000), 16000)
    monkeypatch.setattr(api, "UPLOADS_DIR", tmp_path / "uploads")

    with audio_path.open("rb") as audio_file:
        response = client.post(
            "/api/analyze/standalone",
            files={"audio": ("sample.wav", audio_file, "audio/wav")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["analysis_mode"] == "standalone"
    assert payload["analysis_mode_label"] == "Standalone Analysis"
    assert payload["comparison"]["available"] is False
    assert payload["comparison"]["reason"]
    assert payload["delivery"]["regions"] == []
    assert payload["score"] is None
    assert payload["audio_name"] == "sample.wav"
    assert payload["sample_rate"] == 16000
    assert payload["duration"] > 0
    assert payload["rms"] > 0
    assert list((tmp_path / "uploads").iterdir()) == []


def test_standalone_accepts_stereo_audio_at_different_sample_rate(
    tmp_path,
    monkeypatch,
):
    audio_path = tmp_path / "sample.wav"
    samples = np.column_stack(
        (
            np.linspace(-0.1, 0.1, 22050),
            np.linspace(0.1, -0.1, 22050),
        )
    )
    sf.write(audio_path, samples, 22050)
    monkeypatch.setattr(api, "UPLOADS_DIR", tmp_path / "uploads")

    with audio_path.open("rb") as audio_file:
        response = client.post(
            "/api/analyze/auto",
            files={"audio": ("sample.wav", audio_file, "audio/wav")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["sample_rate"] == 22050
    assert payload["channels"] == 2
    assert payload["duration"] == 1.0
    assert payload["fluency"]["available"] is True


def test_standalone_rejects_corrupt_audio(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "UPLOADS_DIR", tmp_path / "uploads")

    response = client.post(
        "/api/analyze/standalone",
        files={"audio": ("sample.wav", b"not audio", "audio/wav")},
    )

    assert response.status_code == 400
    assert "could not be decoded" in response.json()["detail"]
    assert list((tmp_path / "uploads").iterdir()) == []


def test_standalone_keeps_audio_result_when_coaching_fails(
    tmp_path,
    monkeypatch,
):
    audio_path = tmp_path / "sample.wav"
    sf.write(audio_path, np.linspace(-0.1, 0.1, 16000), 16000)
    monkeypatch.setattr(api, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(
        api,
        "run_fluency_analysis",
        lambda _path: {
            "available": True,
            "model": "V4",
            "detected": True,
            "stutter_probability": 0.8,
            "stutter_probability_percent": 80.0,
            "threshold": 0.67,
            "events": {
                "Block": {
                    "probability": 0.8,
                    "detected": True,
                }
            },
            "possible_events": ["Block"],
        },
    )

    def fail_coaching(_analysis):
        raise RuntimeError("test coaching failure")

    monkeypatch.setattr("rag.coach.generate_coaching", fail_coaching)

    with audio_path.open("rb") as audio_file:
        response = client.post(
            "/api/analyze/standalone",
            files={"audio": ("sample.wav", audio_file, "audio/wav")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["fluency"]["stutter_probability"] == 0.8
    assert payload["coaching"]["status"] == "unavailable"
    assert list((tmp_path / "uploads").iterdir()) == []


def test_existing_contrastive_endpoint_still_analyzes_matching_reference(
    tmp_path,
    monkeypatch,
):
    audio_path = (
        Path(api.ROOT)
        / "data"
        / "contrastive"
        / "audio"
        / "bad"
        / "1272-128104-0000_bad.wav"
    )
    assert audio_path.exists()
    monkeypatch.setattr(api, "UPLOADS_DIR", tmp_path / "uploads")

    with audio_path.open("rb") as audio_file:
        response = client.post(
            "/api/analyze",
            files={
                "audio": (
                    audio_path.name,
                    audio_file,
                    "audio/wav",
                )
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["reference_audio"] == "1272-128104-0000.wav"
    assert "delivery" in payload
    assert "fluency" in payload
